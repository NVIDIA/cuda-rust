/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Static raw-op checks. These cannot prove runtime unsafe preconditions.

use crate::bytecode::Opcode;
use crate::capabilities::{CapabilityError, TargetCapabilities};
use crate::ir::{Attribute, Module, Operation, ScalarType, TileElementType, TileType, Type};
use crate::requirements::Feature;

fn integer(op: &Operation, name: &str) -> Option<i64> {
    op.attributes.iter().find_map(|(n, a)| match a {
        Attribute::Integer(v, _) if n == name => Some(*v),
        _ => None,
    })
}

fn scalar(tile: &TileType) -> Option<ScalarType> {
    match tile.element_type {
        TileElementType::Scalar(s) => Some(s),
        _ => None,
    }
}

fn tile(ty: &Type) -> Option<&TileType> {
    if let Type::Tile(t) = ty.without_pointer_attribute() {
        Some(t)
    } else {
        None
    }
}

fn bits(scalar: ScalarType) -> usize {
    match scalar {
        ScalarType::I1 => 1,
        ScalarType::I4 | ScalarType::F4E2M1FN => 4,
        _ => scalar.byte_width() * 8,
    }
}

fn atomic_mode_valid(mode: Option<i64>, element: ScalarType) -> bool {
    use ScalarType::*;
    match mode {
        Some(0..=3 | 5..=8) => matches!(element, I32 | I64),
        Some(4) => matches!(element, F16 | BF16 | F32 | F64),
        Some(9) => matches!(bits(element), 32 | 64),
        _ => false,
    }
}

fn float_semantics(s: ScalarType) -> Option<(u32, i32, i32)> {
    use ScalarType::*;
    // Precision, minimum normal exponent, maximum normal exponent.
    Some(match s {
        F4E2M1FN => (2, 0, 2),
        F8E4M3FN => (4, -6, 8),
        F8E5M2 => (3, -14, 15),
        F8E8M0FNU => (1, -127, 127),
        F8E5M3FNU => (4, -14, 16),
        F16 => (11, -14, 15),
        BF16 => (8, -126, 127),
        TF32 => (11, -126, 127),
        F32 => (24, -126, 127),
        F64 => (53, -1022, 1023),
        _ => return None,
    })
}

pub(crate) fn verify_type(ty: &Type) -> Result<(), String> {
    let tile_shape = |shape: &[i64], element: Option<ScalarType>| -> Result<(), String> {
        let mut size = 1u64;
        for &dim in shape {
            if dim <= 0 || !(dim as u64).is_power_of_two() {
                return Err("tile dimensions must be positive powers of two".into());
            }
            size = size
                .checked_mul(dim as u64)
                .filter(|n| *n <= 16_777_216)
                .ok_or("tile exceeds 16777216 elements")?;
        }
        if element == Some(ScalarType::F4E2M1FN) && !size.is_multiple_of(2) {
            return Err("f4E2M1FN tiles require an even element count".into());
        }
        Ok(())
    };
    match ty.without_pointer_attribute() {
        Type::Tile(t) => tile_shape(&t.shape, scalar(t)),
        Type::TensorView(v) => {
            if v.shape.len() != v.strides.len() {
                return Err("tensor shape and strides must have matching rank".into());
            }
            if v.shape
                .iter()
                .chain(&v.strides)
                .any(|n| *n <= 0 && *n != crate::ir::DYNAMIC)
            {
                return Err("tensor dimensions and strides must be positive or dynamic".into());
            }
            Ok(())
        }
        Type::PartitionView(_) | Type::StridedView(_) | Type::GatherScatterView(_) => {
            let (shape, tensor, mapping, padding) = match ty.without_pointer_attribute() {
                Type::PartitionView(v) => (
                    &v.tile_shape,
                    &v.tensor_view,
                    Some(&v.dim_map),
                    v.padding_value,
                ),
                Type::StridedView(v) => {
                    if v.traversal_strides.len() != v.tile_shape.len()
                        || v.traversal_strides.iter().any(|n| *n <= 0)
                    {
                        return Err(
                            "strided view requires one positive traversal stride per tile axis"
                                .into(),
                        );
                    }
                    (
                        &v.tile_shape,
                        &v.tensor_view,
                        Some(&v.dim_map),
                        v.padding_value,
                    )
                }
                Type::GatherScatterView(v) => {
                    if v.sparse_dim < 0 || v.sparse_dim as usize >= v.tile_shape.len() {
                        return Err("gather/scatter sparse_dim is outside the tile rank".into());
                    }
                    (&v.tile_shape, &v.tensor_view, None, v.padding_value)
                }
                _ => unreachable!(),
            };
            if shape.is_empty() || shape.len() != tensor.shape.len() {
                return Err("view tile rank must be nonzero and match tensor rank".into());
            }
            if let Some(mapping) = mapping {
                let mut sorted = mapping.clone();
                sorted.sort_unstable();
                if sorted != (0..shape.len() as i32).collect::<Vec<_>>() {
                    return Err("view dim_map must be a permutation of the tensor axes".into());
                }
            }
            if padding.is_some_and(|p| p != crate::ir::PaddingValue::Zero)
                && !tensor.element_type.is_float()
            {
                return Err("nonzero padding modes require a floating-point element type".into());
            }
            verify_type(&Type::TensorView(tensor.clone()))?;
            tile_shape(
                &shape.iter().map(|n| i64::from(*n)).collect::<Vec<_>>(),
                Some(tensor.element_type),
            )
        }
        _ => Ok(()),
    }
}

pub(crate) fn verify_operation(
    module: &Module,
    op: &Operation,
    caps: &TargetCapabilities,
) -> Result<(), CapabilityError> {
    use Opcode::*;
    use ScalarType::*;
    let fail = |why: &str| CapabilityError {
        message: format!("cuda_tile.{}: {why}", op.opcode.name()),
        location: Box::new(op.location.clone()),
    };
    let operand = |i: usize| op.operands.get(i).and_then(|v| tile(module.value_type(*v)));
    let result = || op.result_types.first().and_then(tile);
    let rounding = integer(op, "rounding_mode");
    match op.opcode {
        Bitcast | Reshape | Broadcast | Permute | Pack | Unpack => {
            let src = operand(0).ok_or_else(|| fail("source must be a tile"))?;
            let dst = result().ok_or_else(|| fail("result must be a tile"))?;
            let src_count: i64 = src.shape.iter().product();
            let dst_count: i64 = dst.shape.iter().product();
            let valid = match op.opcode {
                Bitcast => {
                    src.shape == dst.shape
                        && scalar(src)
                            .zip(scalar(dst))
                            .is_some_and(|(a, b)| bits(a) == bits(b))
                }
                Reshape => src.element_type == dst.element_type && src_count == dst_count,
                Broadcast => {
                    src.element_type == dst.element_type
                        && src.shape.len() == dst.shape.len()
                        && src
                            .shape
                            .iter()
                            .zip(&dst.shape)
                            .all(|(a, b)| a == b || *a == 1)
                }
                Permute => {
                    let permutation = op.attributes.iter().find_map(|(n, a)| match a {
                        Attribute::DenseI32Array(p) if n == "permutation" => Some(p),
                        _ => None,
                    });
                    permutation.is_some_and(|p| {
                        let mut sorted = p.clone();
                        sorted.sort_unstable();
                        src.shape.len() >= 2
                            && dst.shape.len() == src.shape.len()
                            && src.element_type == dst.element_type
                            && sorted == (0..src.shape.len() as i32).collect::<Vec<_>>()
                            && p.iter()
                                .enumerate()
                                .all(|(i, n)| dst.shape[i] == src.shape[*n as usize])
                    })
                }
                Pack | Unpack => scalar(src).zip(scalar(dst)).is_some_and(|(a, b)| {
                    let a_bits = bits(a) as i64;
                    let b_bits = bits(b) as i64;
                    src.shape.len() == 1
                        && dst.shape.len() == 1
                        && a_bits != b_bits
                        && src_count * a_bits == dst_count * b_bits
                        && src_count * a_bits % 8 == 0
                }),
                _ => unreachable!(),
            };
            if !valid {
                return Err(fail(
                    "incompatible source/result shapes, element widths or permutation",
                ));
            }
        }
        AtomicRMW | AtomicCAS => {
            let ptr = operand(0).ok_or_else(|| fail("expected pointer tile"))?;
            let val = operand(1).ok_or_else(|| fail("expected value tile"))?;
            let element = scalar(val).ok_or_else(|| fail("atomic value must be numeric"))?;
            if !matches!(&ptr.element_type, TileElementType::Pointer(p) if p.pointee == element)
                || ptr.shape != val.shape
                || result() != Some(val)
            {
                return Err(fail(
                    "atomic pointer, value and result types/shapes must match",
                ));
            }
            if op.opcode == AtomicCAS {
                if operand(2) != Some(val) || !matches!(bits(element), 32 | 64) {
                    return Err(fail(
                        "compare-and-swap requires matching 32-bit or 64-bit values",
                    ));
                }
            } else if !atomic_mode_valid(integer(op, "mode"), element) {
                return Err(fail("atomic mode is not supported for this element type"));
            }
            if !matches!(integer(op, "memory_ordering_semantics"), Some(1..=4))
                || !matches!(integer(op, "memory_scope"), Some(0..=2))
            {
                return Err(fail("atomic operations require relaxed/acquire/release/acq_rel ordering and a valid scope"));
            }
            for v in op
                .operands
                .iter()
                .skip(if op.opcode == AtomicCAS { 3 } else { 2 })
            {
                if let Some(mask) = tile(module.value_type(*v)) {
                    if scalar(mask) != Some(I1) || mask.shape != val.shape {
                        return Err(fail("atomic mask must be an i1 tile with the value shape"));
                    }
                }
            }
        }
        Assume => {
            let value = operand(0);
            let integer_or_pointer =
                value.is_some_and(|t| scalar(t).is_none_or(|s| s.is_integer()));
            for (_, attr) in &op.attributes {
                match attr {
                    Attribute::DivBy(p) => {
                        if !p.divisor.is_power_of_two()
                            || p.divisor > (1u64 << 62)
                            || p.every.is_some() != p.along.is_some()
                        {
                            return Err(fail("div_by requires a power-of-two divisor no greater than 2^62, with every/along both present or absent"));
                        }
                        let tensor_view = op.operands.first().is_some_and(|v| {
                            matches!(
                                module.value_type(*v).without_pointer_attribute(),
                                Type::TensorView(_)
                            )
                        });
                        if !integer_or_pointer && !tensor_view {
                            return Err(fail(
                                "div_by requires an integer/pointer tile or tensor view",
                            ));
                        }
                        if let (Some(every), Some(along)) = (p.every, p.along) {
                            let shape = value.map(|v| &v.shape).ok_or_else(|| {
                                fail("every/along are not allowed on tensor views")
                            })?;
                            if along < 0
                                || along as usize >= shape.len()
                                || every < 0
                                || every > shape[along as usize]
                            {
                                return Err(fail(
                                    "every/along must select a valid axis and group extent",
                                ));
                            }
                        }
                    }
                    Attribute::SameElements(p) => {
                        if !integer_or_pointer {
                            return Err(fail("same_elements requires an integer/pointer tile"));
                        }
                        let shape = &value.unwrap().shape;
                        if p.values.len() != shape.len()
                            || p.values.iter().zip(shape).any(|(n, dim)| *n < 0 || n > dim)
                        {
                            return Err(fail("same_elements requires one group size per axis, between zero and that axis's extent"));
                        }
                    }
                    Attribute::Bounded(p) => {
                        let scalar = value
                            .and_then(scalar)
                            .filter(|s| s.is_integer())
                            .ok_or_else(|| fail("bounded requires an integer tile"))?;
                        let bits = match scalar {
                            I1 => 1,
                            I4 => 4,
                            _ => scalar.byte_width() * 8,
                        };
                        let (min, max) = if bits == 64 {
                            (i64::MIN, i64::MAX)
                        } else {
                            (-(1i64 << (bits - 1)), (1i64 << (bits - 1)) - 1)
                        };
                        if p.lb.into_iter().chain(p.ub).any(|v| v < min || v > max)
                            || matches!((p.lb, p.ub), (Some(l), Some(u)) if l > u)
                        {
                            return Err(fail("bounded endpoints must be ordered and fit the signed operand range"));
                        }
                    }
                    _ => {}
                }
            }
        }
        MakeTensorView | MakePartitionView | MakeGatherScatterView | MakeStridedView => {
            let source = op.operands.first().map(|v| module.value_type(*v));
            if source.and_then(Type::pointer_attribute)
                != op.result_types.first().and_then(Type::pointer_attribute)
            {
                return Err(fail(
                    "view must preserve both presence and value of the base pointer's ptr_attr",
                ));
            }
        }
        Alloca => {
            let count = integer(op, "num_elem")
                .ok_or_else(|| fail("num_elem must be a compile-time integer"))?;
            let alignment = integer(op, "alignment").unwrap_or(0);
            let natural = result()
                .and_then(|t| match &t.element_type {
                    TileElementType::Pointer(p) => Some(p.pointee.byte_width() as i64),
                    _ => None,
                })
                .ok_or_else(|| fail("alloca result must be a pointer tile"))?;
            if count < 0 {
                return Err(fail("num_elem must be nonnegative"));
            }
            if alignment <= 0 || !(alignment as u64).is_power_of_two() || alignment < natural {
                return Err(fail(
                    "alignment must be a nonzero power of two at least the element's natural size",
                ));
            }
        }
        FPowI => {
            let base = operand(0).ok_or_else(|| fail("base must be a floating-point tile"))?;
            let exponent = operand(1).ok_or_else(|| fail("exponent must be an integer tile"))?;
            if !matches!(scalar(base), Some(F16 | BF16 | F32 | F64)) || result() != Some(base) {
                return Err(fail(
                    "base and result must be matching f16, bf16, f32 or f64 tiles",
                ));
            }
            if base.shape != exponent.shape
                || !matches!(scalar(exponent), Some(I1 | I8 | I16 | I32))
            {
                return Err(fail("exponent must have the base's shape and element type i1, i8, i16 or i32 (interpreted as signed)"));
            }
        }
        MmaF | MmaI | MmaFScaled => {
            let a = operand(0).ok_or_else(|| fail("MMA lhs must be a tile"))?;
            let b = operand(1).ok_or_else(|| fail("MMA rhs must be a tile"))?;
            let c = operand(2).ok_or_else(|| fail("MMA accumulator must be a tile"))?;
            let rank = a.shape.len();
            if !matches!(rank, 2 | 3)
                || b.shape.len() != rank
                || c.shape.len() != rank
                || result() != Some(c)
            {
                return Err(fail(
                    "MMA requires matching 2D or 3D ranks and identical accumulator/result types",
                ));
            }
            let (row, col) = (rank - 2, rank - 1);
            if a.shape[row] != c.shape[row]
                || b.shape[col] != c.shape[col]
                || a.shape[col] != b.shape[row]
                || (rank == 3 && (a.shape[0] != b.shape[0] || a.shape[0] != c.shape[0]))
            {
                return Err(fail("MMA batch, M, N and K dimensions must match"));
            }
            let input = scalar(a);
            let acc = scalar(c);
            if input != scalar(b) {
                return Err(fail("MMA lhs and rhs element types must match"));
            }
            if op.opcode == MmaF {
                let valid = match input {
                    Some(F4E2M1FN | F8E4M3FN | F8E5M2 | F16) => matches!(acc, Some(F16 | F32)),
                    Some(BF16 | TF32 | F32) => acc == Some(F32),
                    Some(F64) => acc == Some(F64),
                    _ => false,
                };
                if !valid {
                    return Err(fail(
                        "unsupported MMA input/accumulator element type combination",
                    ));
                }
            } else if op.opcode == MmaFScaled {
                let sa = operand(3).ok_or_else(|| fail("scaled MMA requires lhs scale tile"))?;
                let sb = operand(4).ok_or_else(|| fail("scaled MMA requires rhs scale tile"))?;
                if sa.shape.len() != rank
                    || sb.shape.len() != rank
                    || acc != Some(F32)
                    || scalar(sa) != scalar(sb)
                {
                    return Err(fail(
                        "scaled MMA requires matching scale ranks/types and f32 accumulation",
                    ));
                }
                if sa.shape[row] != a.shape[row]
                    || sb.shape[col] != b.shape[col]
                    || sa.shape[col] != sb.shape[row]
                    || (rank == 3 && (sa.shape[0] != a.shape[0] || sb.shape[0] != b.shape[0]))
                    || sa.shape[col] <= 0
                    || a.shape[col] % sa.shape[col] != 0
                {
                    return Err(fail(
                        "scaled MMA scale shapes must match batch/M/N and divide K",
                    ));
                }
                let vector = a.shape[col] / sa.shape[col];
                let valid = match (input, scalar(sa)) {
                    (Some(F4E2M1FN), Some(F8E4M3FN)) => vector == 16,
                    (Some(F4E2M1FN), Some(F8E8M0FNU | F8E5M3FNU)) => matches!(vector, 16 | 32),
                    (Some(F8E4M3FN | F8E5M2), Some(F8E8M0FNU)) => vector == 32,
                    _ => false,
                };
                if !valid {
                    return Err(fail(
                        "unsupported scaled MMA element types or scale-vector size",
                    ));
                }
            }
        }
        Insert | Extract => {
            let source = operand(0).ok_or_else(|| fail("source must be a tile"))?;
            let output = result().ok_or_else(|| fail("result must be a tile"))?;
            let (small, large, offset) = if op.opcode == Insert {
                if operand(1) != Some(output) {
                    return Err(fail("destination and result must have identical types"));
                }
                (source, output, 2)
            } else {
                (output, source, 1)
            };
            if small.element_type != large.element_type
                || small.shape.len() != large.shape.len()
                || small
                    .shape
                    .iter()
                    .zip(&large.shape)
                    .any(|(s, l)| *s <= 0 || l % s != 0)
            {
                return Err(fail("source and result must have matching element types and rank, with divisible tile dimensions"));
            }
            if op.operands.len() != offset + small.shape.len()
                || op.operands.iter().skip(offset).any(|v| !matches!(tile(module.value_type(*v)), Some(t) if t.shape.is_empty() && scalar(t) == Some(I32))) {
                return Err(fail("expected one scalar i32 index per tile dimension"));
            }
        }
        FToI => {
            if rounding != Some(6) {
                return Err(fail("rounding_mode must be nearest_int_to_zero"));
            }
        }
        FToF => {
            let src = operand(0)
                .and_then(scalar)
                .ok_or_else(|| fail("source must be a floating-point tile"))?;
            let dst = result()
                .and_then(scalar)
                .ok_or_else(|| fail("result must be a floating-point tile"))?;
            let (sp, smin, smax) =
                float_semantics(src).ok_or_else(|| fail("source must be floating point"))?;
            let (dp, dmin, dmax) =
                float_semantics(dst).ok_or_else(|| fail("result must be floating point"))?;
            if src == dst {
                return Err(fail("float-to-float conversion must change element type"));
            }
            let r = rounding.unwrap_or(-1);
            let allowed = if dst == F8E8M0FNU {
                matches!(r, 1 | 3)
            } else if caps.bytecode_version < Feature::ExtendedFloatRounding.since()
                || dst.byte_width() < 2
            {
                r == 0
            } else if dp >= sp && dmin <= smin && dmax >= smax {
                matches!(r, 0..=3 | 7)
            } else if src == F64 && dst == F32 {
                matches!(r, 0..=3)
            } else if src == F32 && dst == TF32 {
                matches!(r, 0 | 1 | 7)
            } else {
                matches!(r, 0 | 1)
            };
            if !allowed {
                return Err(fail(&format!(
                    "rounding_mode {r} is not supported for {src:?} to {dst:?} in Tile IR {}",
                    caps.bytecode_version
                )));
            }
        }
        AddF | SubF | MulF | Fma => {
            if !matches!(rounding, Some(0..=3)) {
                return Err(fail(
                    "rounding_mode must be nearest_even, positive_inf, negative_inf or zero",
                ));
            }
        }
        Exp | TanH => {
            let r = rounding.unwrap_or(5);
            if !matches!(r, 4 | 5) {
                return Err(fail("rounding_mode must be approx or full"));
            }
            if r == 4 && operand(0).and_then(scalar) != Some(F32) {
                return Err(fail("approx rounding requires f32"));
            }
        }
        GdcLaunchDependentsTko | GdcWaitTko | MemoryFenceAliasTko => {
            if op.result_types != [Type::Token]
                || op
                    .operands
                    .iter()
                    .any(|v| module.value_type(*v) != &Type::Token)
                || op.operands.len() > 1
                || (op.opcode == MemoryFenceAliasTko && op.operands.len() != 1)
            {
                return Err(fail("invalid token operand/result signature"));
            }
        }
        LoadViewTko | StoreViewTko | AtomicRedViewTko => {
            let view = op
                .operands
                .get(usize::from(op.opcode == StoreViewTko))
                .map(|v| module.value_type(*v))
                .ok_or_else(|| fail("expected view operand"))?;
            let (shape, element, padding, sparse) = match view.without_pointer_attribute() {
                Type::PartitionView(v) => (
                    &v.tile_shape,
                    v.tensor_view.element_type,
                    v.padding_value,
                    None,
                ),
                Type::StridedView(v) => (
                    &v.tile_shape,
                    v.tensor_view.element_type,
                    v.padding_value,
                    None,
                ),
                Type::GatherScatterView(v) => (
                    &v.tile_shape,
                    v.tensor_view.element_type,
                    v.padding_value,
                    Some(v.sparse_dim),
                ),
                _ => return Err(fail("expected a partition, strided or gather/scatter view")),
            };
            let is_load = op.opcode == LoadViewTko;
            let atomic = op.opcode == AtomicRedViewTko;
            let index_start = if op.opcode == StoreViewTko { 2 } else { 1 };
            let value = if is_load {
                result()
            } else if atomic {
                operand(1 + shape.len())
            } else {
                operand(0)
            };
            let value = value.ok_or_else(|| fail("expected loaded/stored tile"))?;
            if value.shape != shape.iter().map(|&n| i64::from(n)).collect::<Vec<_>>()
                || scalar(value) != Some(element)
            {
                return Err(fail(
                    "view and value must have matching tile shape and element type",
                ));
            }
            if atomic && (sparse.is_some() || padding.is_some() || integer(op, "mode") == Some(9)) {
                return Err(fail("view atomic reduction requires an unpadded partition or strided view and a non-exchange mode"));
            }
            if atomic && !atomic_mode_valid(integer(op, "mode"), element) {
                return Err(fail(
                    "atomic reduction mode is not supported for this element type",
                ));
            }
            let segments = op.attributes.iter().find_map(|(n, a)| {
                if n == "operandSegmentSizes" {
                    if let Attribute::Array(v) = a {
                        Some(v)
                    } else {
                        None
                    }
                } else {
                    None
                }
            });
            let count = segments.and_then(|s| s.get(if is_load || atomic { 1 } else { 2 }));
            if !matches!(count, Some(Attribute::Integer(n, _)) if *n == shape.len() as i64) {
                return Err(fail("expected one index per view dimension"));
            }
            let mut index_element = None;
            for axis in 0..shape.len() {
                let index = operand(index_start + axis)
                    .ok_or_else(|| fail("indices must be integer tiles"))?;
                let element = scalar(index)
                    .filter(|s| s.is_integer())
                    .ok_or_else(|| fail("indices must be integer tiles"))?;
                if sparse == Some(axis as i32) {
                    if index.shape != [i64::from(shape[axis])] {
                        return Err(fail(
                            "sparse index must be a 1D tile with the sparse axis's tile extent",
                        ));
                    }
                } else {
                    if !index.shape.is_empty() {
                        return Err(fail("dense indices must be scalar integer tiles"));
                    }
                    if index_element.is_some_and(|s| s != element) {
                        return Err(fail("dense index element types must match"));
                    }
                    index_element = Some(element);
                }
            }
            let order = integer(op, "memory_ordering_semantics").unwrap_or(0);
            let scope = integer(op, "memory_scope");
            if (is_load && !matches!(order, 0..=2))
                || (!is_load && !matches!(order, 0 | 1 | 3))
                || (atomic && (order != 1 || !matches!(scope, Some(0 | 1))))
                || (order == 0 && scope.is_some())
                || (order != 0 && !matches!(scope, Some(0..=2)))
            {
                return Err(fail(
                    "invalid memory ordering/scope combination for this view operation",
                ));
            }
        }
        _ => {}
    }
    if op
        .attributes
        .iter()
        .any(|(n, a)| n == "flush_to_zero" && !matches!(a, Attribute::Bool(false)))
        && operand(0).and_then(scalar) != Some(F32)
    {
        return Err(fail("flush_to_zero requires f32"));
    }
    Ok(())
}
