/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Static raw-op checks. These cannot prove runtime unsafe preconditions.

use std::collections::HashSet;

use crate::bytecode::Opcode;
use crate::capabilities::{CapabilityError, TargetCapabilities};
use crate::ir::{
    Access, Attribute, ConstParam, FragmentType, FuncType, Layout, MemorySpace, Module, Operation,
    PreviewParamType, ScalarType, ScratchRequest, SymExpr, Synchronization, TileElementType,
    TileType, Type, ACCESS_ATTR,
};
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
        Type::Fragment(fragment) => verify_fragment(fragment),
        _ => Ok(()),
    }
}

// =========================================================================
// Public-preview SIMT interop structure (PreviewOps.cpp, Types.cpp)
// =========================================================================

/// Product of the integer leaves of a concrete (possibly nested) mode.
/// `None` for a symbolic leaf or an overflowing product.
pub(crate) fn mode_size(mode: &SymExpr) -> Option<i64> {
    match mode {
        SymExpr::Tuple(values) => values
            .iter()
            .try_fold(1i64, |acc, v| acc.checked_mul(mode_size(v)?)),
        other => other.as_int(),
    }
}

/// Product of the shape leaves whose stride is nonzero (`cute::filter_zeros`):
/// a stride-0 leaf is a broadcast and does not consume tile elements.
pub(crate) fn injective_mode_size(shape: &SymExpr, stride: &SymExpr) -> Option<i64> {
    match (shape, stride) {
        (SymExpr::Tuple(shapes), SymExpr::Tuple(strides)) if shapes.len() == strides.len() => {
            shapes.iter().zip(strides).try_fold(1i64, |acc, (s, d)| {
                acc.checked_mul(injective_mode_size(s, d)?)
            })
        }
        (SymExpr::Tuple(_), _) | (_, SymExpr::Tuple(_)) => None,
        (shape, stride) => {
            let (shape, stride) = (shape.as_int()?, stride.as_int()?);
            Some(if stride == 0 { 1 } else { shape })
        }
    }
}

/// Whether every non-unit leaf of a concrete mode has stride zero.
pub(crate) fn mode_has_zero_stride(shape: &SymExpr, stride: &SymExpr) -> Option<bool> {
    match (shape, stride) {
        (SymExpr::Tuple(shapes), SymExpr::Tuple(strides)) if shapes.len() == strides.len() => {
            shapes
                .iter()
                .zip(strides)
                .try_fold(true, |acc, (s, d)| Some(acc && mode_has_zero_stride(s, d)?))
        }
        (SymExpr::Tuple(_), _) | (_, SymExpr::Tuple(_)) => None,
        (shape, stride) => {
            let (shape, stride) = (shape.as_int()?, stride.as_int()?);
            Some(shape == 1 || stride == 0)
        }
    }
}

/// `verifySymbolicExprKind(Int)`: an integer context (a layout leaf, an
/// alignment, a scratch size, an index argument) takes integer literals,
/// variables, context builtins and arithmetic over them at any depth.
/// Tuples, strings, types and the toward-zero `Div` have no meaning there.
pub(crate) fn verify_int_expr(expr: &SymExpr, what: &str) -> Result<(), String> {
    match expr {
        SymExpr::Int(_) | SymExpr::Var(_) | SymExpr::Builtin { .. } => Ok(()),
        SymExpr::Tuple(_) => Err(format!("{what} must be an integer expression, got a tuple")),
        SymExpr::ConstStr(_) => Err(format!(
            "{what} must be an integer expression, got a string constant"
        )),
        SymExpr::ConcreteType(_) => {
            Err(format!("{what} must be an integer expression, got a type"))
        }
        SymExpr::Div(..) => Err(format!(
            "{what} must be an integer expression; toward-zero division is unsupported in \
             integer contexts, use floor or ceiling division"
        )),
        SymExpr::Mul(l, r)
        | SymExpr::Add(l, r)
        | SymExpr::Sub(l, r)
        | SymExpr::FloorDiv(l, r)
        | SymExpr::CeilDiv(l, r) => {
            verify_int_expr(l, what)?;
            verify_int_expr(r, what)
        }
        SymExpr::Neg(x) => verify_int_expr(x, what),
    }
}

/// `verifySymbolicExprKind(Tuple)`: a shape is a variable standing for the
/// whole shape, or a tuple whose elements are integer expressions or nested
/// tuples.
fn verify_tuple_expr(expr: &SymExpr, what: &str) -> Result<(), String> {
    match expr {
        SymExpr::Var(_) => Ok(()),
        SymExpr::Tuple(values) => values.iter().try_for_each(|value| match value {
            SymExpr::Tuple(_) => verify_tuple_expr(value, what),
            _ => verify_int_expr(value, what),
        }),
        _ => Err(format!(
            "{what} must be a tuple expression (a tuple or a shape variable)"
        )),
    }
}

/// `verifyConstantArgumentKind`: a constant argument has the kind of the
/// parameter it binds. A variable forwards an in-scope parameter and is
/// accepted for every kind; its own kind is checked where it is declared.
pub(crate) fn verify_const_arg_kind(
    index: usize,
    param: &ConstParam,
    arg: &SymExpr,
) -> Result<(), String> {
    let what = format!(
        "constant argument #{index} for constant parameter '{}'",
        param.name
    );
    match &param.ty {
        Type::Index => verify_int_expr(arg, &what),
        Type::PreviewParam(PreviewParamType::Shape) => verify_tuple_expr(arg, &what),
        Type::PreviewParam(PreviewParamType::DType) => match arg {
            SymExpr::Var(_) | SymExpr::ConcreteType(_) => Ok(()),
            _ => Err(format!(
                "{what} must be a datatype expression (a concrete type or a variable)"
            )),
        },
        Type::PreviewParam(PreviewParamType::String) => match arg {
            SymExpr::Var(_) | SymExpr::ConstStr(_) => Ok(()),
            _ => Err(format!(
                "{what} must be a string expression (a string constant or a variable)"
            )),
        },
        _ => Err(format!(
            "constant parameter '{}' must be index, preview$shape, preview$Datatype or \
             preview$string",
            param.name
        )),
    }
}

/// The first variable of `expr` missing from `declared`; with an empty set,
/// the first variable at all.
fn undeclared_variable<'a>(expr: &'a SymExpr, declared: &HashSet<&str>) -> Option<&'a str> {
    match expr {
        SymExpr::Var(name) => (!declared.contains(name.as_str())).then_some(name.as_str()),
        SymExpr::Tuple(values) => values
            .iter()
            .find_map(|value| undeclared_variable(value, declared)),
        SymExpr::Mul(l, r)
        | SymExpr::Div(l, r)
        | SymExpr::Add(l, r)
        | SymExpr::Sub(l, r)
        | SymExpr::FloorDiv(l, r)
        | SymExpr::CeilDiv(l, r) => {
            undeclared_variable(l, declared).or_else(|| undeclared_variable(r, declared))
        }
        SymExpr::Neg(x) => undeclared_variable(x, declared),
        SymExpr::Int(_)
        | SymExpr::Builtin { .. }
        | SymExpr::ConstStr(_)
        | SymExpr::ConcreteType(_) => None,
    }
}

/// Names of the non-wildcard parameters: the only variables an
/// implementation's fragments and scratch request may reference.
pub(crate) fn declared_variables<'a>(
    params: impl IntoIterator<Item = &'a ConstParam>,
) -> HashSet<&'a str> {
    params
        .into_iter()
        .filter(|p| !p.is_wildcard())
        .map(|p| p.name.as_str())
        .collect()
}

/// `verifyNativeImpl`: every variable in a fragment's `mem_layout`, `fragmap`
/// and `alignment` must be a declared parameter. The static shape cannot
/// hold one, since `verify_fragment` rejects symbolic dimensions.
pub(crate) fn verify_fragment_variables(
    fragment: &FragmentType,
    declared: &HashSet<&str>,
) -> Result<(), String> {
    fn leaves(layout: &Layout) -> Vec<&SymExpr> {
        layout.shapes.iter().chain(&layout.strides).collect()
    }
    let fields = [
        ("mem_layout", leaves(&fragment.mem_layout)),
        ("fragmap", leaves(&fragment.fragmap)),
        ("alignment", vec![&fragment.alignment]),
    ];
    for (field, exprs) in fields {
        if let Some(name) = exprs
            .iter()
            .find_map(|expr| undeclared_variable(expr, declared))
        {
            return Err(format!(
                "field {field} references an undeclared symbolic variable '{name}'; it must be \
                 a named constant parameter of the implementation"
            ));
        }
    }
    Ok(())
}

/// A scratch request may only reference the implementation's declared
/// parameters (and context builtins).
pub(crate) fn verify_scratch_variables(
    scratch: &ScratchRequest,
    declared: &HashSet<&str>,
) -> Result<(), String> {
    for (expr, what) in [
        (&scratch.size, "scratch size"),
        (&scratch.alignment, "scratch alignment"),
    ] {
        if let Some(name) = undeclared_variable(expr, declared) {
            return Err(format!(
                "{what} references an undeclared symbolic variable '{name}'; it must be a named \
                 constant parameter of the implementation"
            ));
        }
    }
    Ok(())
}

/// Shapes and strides must nest identically; concrete extents must be
/// positive and concrete strides non-negative (`LayoutAttr::verify`).
fn verify_layout(layout: &Layout, what: &str) -> Result<(), String> {
    if layout.shapes.len() != layout.strides.len() {
        return Err(format!(
            "{what} shapes and strides must have the same rank, got {} vs {}",
            layout.shapes.len(),
            layout.strides.len()
        ));
    }
    fn mode(shape: &SymExpr, stride: &SymExpr, what: &str) -> Result<(), String> {
        match (shape, stride) {
            (SymExpr::Tuple(s), SymExpr::Tuple(d)) => {
                if s.len() != d.len() {
                    return Err(format!(
                        "{what} shapes and strides must have the same nesting structure"
                    ));
                }
                s.iter().zip(d).try_for_each(|(s, d)| mode(s, d, what))
            }
            (SymExpr::Tuple(_), _) | (_, SymExpr::Tuple(_)) => Err(format!(
                "{what} shapes and strides must have the same nesting structure"
            )),
            (shape, stride) => {
                verify_int_expr(shape, &format!("{what} shape leaf"))?;
                verify_int_expr(stride, &format!("{what} stride leaf"))?;
                if shape.as_int().is_some_and(|n| n <= 0) {
                    return Err(format!("{what} shape extents must be positive"));
                }
                if stride.as_int().is_some_and(|n| n < 0) {
                    return Err(format!("{what} strides must be non-negative"));
                }
                Ok(())
            }
        }
    }
    layout
        .shapes
        .iter()
        .zip(&layout.strides)
        .try_for_each(|(s, d)| mode(s, d, what))
}

/// Flatten a concrete layout into `(extent, stride)` leaves, or `None` when
/// a leaf is symbolic.
fn concrete_leaves(layout: &Layout) -> Option<Vec<(i64, i64)>> {
    fn walk(shape: &SymExpr, stride: &SymExpr, out: &mut Vec<(i64, i64)>) -> Option<()> {
        match (shape, stride) {
            (SymExpr::Tuple(s), SymExpr::Tuple(d)) => {
                s.iter().zip(d).try_for_each(|(s, d)| walk(s, d, out))
            }
            (SymExpr::Tuple(_), _) | (_, SymExpr::Tuple(_)) => None,
            (s, d) => {
                out.push((s.as_int()?, d.as_int()?));
                Some(())
            }
        }
    }
    let mut out = Vec::new();
    layout
        .shapes
        .iter()
        .zip(&layout.strides)
        .try_for_each(|(s, d)| walk(s, d, &mut out))?;
    Some(out)
}

/// A fragment buffer holds a single invocation's footprint and the ABI
/// hands over a bare pointer, so `mem_layout` must be a permutation of
/// `[0, count)`. Checked only once the layout and shape are concrete.
pub(crate) fn verify_mem_layout(layout: &Layout, count: i64) -> Result<(), String> {
    let Some(leaves) = concrete_leaves(layout) else {
        return Ok(());
    };
    let domain = leaves
        .iter()
        .try_fold(1i64, |acc, (n, _)| acc.checked_mul(*n))
        .ok_or("mem_layout domain size must be representable as a signed 64-bit integer")?;
    if domain != count {
        return Err(format!(
            "mem_layout domain size ({domain}) must equal the fragment element count ({count})"
        ));
    }
    let cosize = leaves
        .iter()
        .try_fold(1i64, |acc, (n, d)| {
            acc.checked_add((n - 1).checked_mul(*d)?)
        })
        .ok_or("mem_layout storage span must be representable as a signed 64-bit integer")?;
    if cosize != count {
        return Err(format!(
            "mem_layout must be a bijection on the footprint elements (a permutation of \
             [0, {count})), but its codomain spans {cosize} element(s)"
        ));
    }
    // Enumerate the offsets for small footprints; the size and span checks
    // above already bound larger ones.
    if count <= (1 << 20) && count > 0 {
        let mut seen = vec![false; count as usize];
        let mut index = vec![0i64; leaves.len()];
        loop {
            let offset: i64 = index.iter().zip(&leaves).map(|(i, (_, d))| i * d).sum();
            if offset < 0 || offset >= count || seen[offset as usize] {
                return Err(
                    "mem_layout must be a bijection on the footprint elements, but its \
                     modes overlap or leave gaps"
                        .into(),
                );
            }
            seen[offset as usize] = true;
            let mut carry = true;
            for (i, (n, _)) in index.iter_mut().zip(&leaves) {
                if carry {
                    *i += 1;
                    carry = *i == *n;
                    if carry {
                        *i = 0;
                    }
                }
            }
            if carry {
                break;
            }
        }
    }
    Ok(())
}

/// An alignment is one integer expression; a concrete value must be a
/// positive power of two.
fn verify_alignment(alignment: &SymExpr, what: &str) -> Result<(), String> {
    verify_int_expr(alignment, what)?;
    if let Some(value) = alignment.as_int() {
        if value <= 0 || !(value as u64).is_power_of_two() {
            return Err(format!(
                "{what} must be a positive power of two, got {value}"
            ));
        }
    }
    Ok(())
}

/// `FragmentType::verify`: static positive shape, integer power-of-two
/// alignment, well-formed layouts, a three-mode fragmap and a permutation
/// `mem_layout`.
pub(crate) fn verify_fragment(fragment: &FragmentType) -> Result<(), String> {
    for (i, dim) in fragment.shape.iter().enumerate() {
        if *dim == crate::ir::DYNAMIC {
            return Err(format!(
                "fragment shape dimension {i} is symbolic; this writer supports concrete \
                 fragment shapes only"
            ));
        }
        if *dim <= 0 {
            return Err(format!(
                "fragment shape dimension {i} must be positive, got {dim}"
            ));
        }
    }
    let count = fragment
        .num_elements()
        .ok_or("fragment element count must fit in a signed 64-bit integer")?;
    verify_alignment(&fragment.alignment, "fragment alignment")?;
    verify_layout(&fragment.mem_layout, "mem_layout")?;
    verify_layout(&fragment.fragmap, "fragmap")?;
    if fragment.fragmap.rank() != 3 {
        return Err(format!(
            "fragmap must have 3 modes (invocation, participant, footprint), got {}",
            fragment.fragmap.rank()
        ));
    }
    verify_mem_layout(&fragment.mem_layout, count)
}

/// `verifySynchronizationParticipation`: warp needs a positive multiple of
/// 32 participants, warpgroup of 128. `cta` is checked by the consumer,
/// which knows the calling kernel's thread count.
pub(crate) fn verify_synchronization_participation(
    synchronization: Synchronization,
    participants: i64,
) -> Result<(), String> {
    let (granularity, scope) = match synchronization {
        Synchronization::None | Synchronization::Cta => return Ok(()),
        Synchronization::Warp => (32, "warp"),
        Synchronization::Warpgroup => (128, "warpgroup"),
    };
    if participants <= 0 || participants % granularity != 0 {
        return Err(format!(
            "synchronization '{scope}' requires the fragmap participant mode size \
             ({participants}) to be a positive multiple of {granularity}"
        ));
    }
    Ok(())
}

/// `out`/`inout` arguments must form one contiguous run with no `in`
/// argument between them. Upstream applies this to native implementations
/// only (PreviewOps.cpp, `verifyNativeImpl`); `verify_preview_func` applies
/// it to declarations as well, as this crate's narrowing.
pub(crate) fn verify_access_run(accesses: &[Access]) -> Result<(), String> {
    let non_in: Vec<usize> = accesses
        .iter()
        .enumerate()
        .filter(|(_, a)| **a != Access::In)
        .map(|(i, _)| i)
        .collect();
    if let (Some(first), Some(last)) = (non_in.first(), non_in.last()) {
        if last - first + 1 != non_in.len() {
            return Err(format!(
                "out/inout arguments must form a single contiguous run; found {} out/inout \
                 argument(s) spanning indices {first}..{last} with an 'in' argument in between",
                non_in.len()
            ));
        }
    }
    Ok(())
}

/// A concrete scratch size is non-negative and a concrete alignment a
/// positive power of two (`ScratchRequestAttr::verify`).
pub(crate) fn verify_scratch(scratch: &ScratchRequest) -> Result<(), String> {
    verify_int_expr(&scratch.size, "scratch size")?;
    if scratch.size.as_int().is_some_and(|n| n < 0) {
        return Err("scratch size must be non-negative".into());
    }
    verify_alignment(&scratch.alignment, "scratch alignment")
}

/// The `cuda_tile.preview$access` of every argument, from an `arg_attrs`
/// array of dictionaries.
pub(crate) fn argument_accesses(
    arg_attrs: Option<&Attribute>,
    num_args: usize,
) -> Result<Vec<Access>, String> {
    let dicts = match arg_attrs {
        Some(Attribute::Array(dicts)) if dicts.len() == num_args => dicts,
        _ => {
            return Err(format!(
                "every argument needs a '{ACCESS_ATTR}' attribute: expected an arg_attrs \
                 array of {num_args} dictionaries"
            ))
        }
    };
    dicts
        .iter()
        .enumerate()
        .map(|(i, dict)| match dict {
            Attribute::Dictionary(entries) => entries
                .iter()
                .find_map(|(k, v)| match v {
                    Attribute::Access(a) if k == ACCESS_ATTR => Some(*a),
                    _ => None,
                })
                .ok_or_else(|| format!("argument {i} must carry a '{ACCESS_ATTR}' attribute")),
            _ => Err(format!("arg_attrs[{i}] must be a dictionary")),
        })
        .collect()
}

/// Constant parameters have unique, non-empty names and a parameter type;
/// the wildcard `?` is only meaningful on an implementation.
pub(crate) fn verify_const_param_list<'a>(
    params: impl IntoIterator<Item = &'a ConstParam>,
    allow_wildcard: bool,
) -> Result<(), String> {
    let mut names = HashSet::new();
    for (i, param) in params.into_iter().enumerate() {
        if param.is_wildcard() {
            if allow_wildcard {
                continue;
            }
            return Err(format!(
                "const_params[{i}]: '?' wildcard is only valid in extern_impl const_params"
            ));
        }
        if param.name.is_empty() {
            return Err("constant parameter name cannot be empty".into());
        }
        if !matches!(param.ty, Type::Index | Type::PreviewParam(_)) {
            return Err(format!(
                "constant parameter '{}' must be index, preview$shape, preview$Datatype or \
                 preview$string",
                param.name
            ));
        }
        if !names.insert(param.name.as_str()) {
            return Err(format!(
                "duplicate constant parameter name: '{}'",
                param.name
            ));
        }
    }
    Ok(())
}

/// The `ConstParam`s of a `const_params` array attribute.
fn const_params(params: Option<&Attribute>) -> Result<Vec<&ConstParam>, String> {
    let Some(params) = params else {
        return Ok(Vec::new());
    };
    let Attribute::Array(params) = params else {
        return Err("const_params must be an array".into());
    };
    params
        .iter()
        .enumerate()
        .map(|(i, param)| match param {
            Attribute::ConstParam(param) => Ok(param),
            _ => Err(format!("const_params[{i}] must be a constant parameter")),
        })
        .collect()
}

fn function_type(op: &Operation) -> Result<&FuncType, String> {
    match op.attributes.iter().find(|(k, _)| k == "function_type") {
        Some((_, Attribute::Type(Type::Func(f)))) => Ok(f),
        _ => Err("function_type must be a function type".into()),
    }
}

fn attr<'a>(op: &'a Operation, name: &str) -> Option<&'a Attribute> {
    op.attributes
        .iter()
        .find_map(|(k, v)| if k == name { Some(v) } else { None })
}

/// A symbol-valued attribute (`sym_name`, `implements`, `callee`): a
/// non-empty string.
fn symbol<'a>(op: &'a Operation, name: &str) -> Result<&'a str, String> {
    match attr(op, name) {
        Some(Attribute::String(s)) if !s.is_empty() => Ok(s),
        _ => Err(format!("{name} must be a non-empty symbol name")),
    }
}

/// The structural rules of the three public-preview ops, shared by
/// `verify_operation` and `verify_library`, so the typed library layer runs
/// the same checks on the modules it builds instead of restating them.
/// Other opcodes pass.
pub(crate) fn verify_preview_op(module: &Module, op: &Operation) -> Result<(), String> {
    match op.opcode {
        Opcode::PreviewFunc => verify_preview_func(op),
        Opcode::PreviewExternImpl => verify_extern_impl(op),
        Opcode::PreviewCall => verify_preview_call(module, op),
        _ => Ok(()),
    }
}

/// `preview$func extern`: tile-typed boundary with per-argument access, a
/// contiguous out/inout run, and results equal to the out/inout tiles in
/// order. Two of these are this crate's narrowings of upstream's
/// `FuncOp::verify`, which the SIMT implementation path needs: upstream
/// tolerates extra trailing results after the bound outputs, but there is
/// no fragment to bind an extra result to; and upstream requires the
/// contiguous out/inout run only of native implementations
/// (`verifyNativeImpl`), but every declaration here is implemented by one,
/// whose arguments mirror the declaration's, so the order is checked where
/// it is chosen. Only declarations are supported.
fn verify_preview_func(op: &Operation) -> Result<(), String> {
    if attr(op, "is_extern") != Some(&Attribute::Bool(true)) {
        return Err(
            "only extern declarations are supported; a body-bearing preview$func cannot \
                    be written"
                .into(),
        );
    }
    symbol(op, "sym_name")?;
    let func = function_type(op)?;
    if func.inputs.is_empty() {
        return Err("extern functions must declare at least one input".into());
    }
    for (i, ty) in func.inputs.iter().enumerate() {
        if !matches!(ty, Type::Tile(_)) {
            return Err(format!("input type {i} must be a cuda_tile.tile"));
        }
        verify_type(ty).map_err(|e| format!("input type {i}: {e}"))?;
    }
    for (i, ty) in func.results.iter().enumerate() {
        if !matches!(ty, Type::Tile(_)) {
            return Err(format!("result type {i} must be a cuda_tile.tile"));
        }
    }
    let accesses = argument_accesses(attr(op, "arg_attrs"), func.inputs.len())?;
    verify_access_run(&accesses).map_err(|e| {
        format!(
            "{e}; this is this crate's narrowing on declarations (upstream FuncOp::verify \
             does not require the run, its native implementation verifier does), because \
             every declaration here is implemented by a native SIMT function"
        )
    })?;
    let bound: Vec<&Type> = func
        .inputs
        .iter()
        .zip(&accesses)
        .filter(|(_, a)| **a != Access::In)
        .map(|(ty, _)| ty)
        .collect();
    if bound.len() != func.results.len() || bound.iter().zip(&func.results).any(|(b, r)| *b != r) {
        return Err(
            "results must be the out/inout argument tile types in order and nothing else; \
             upstream tolerates extra trailing results, which the SIMT implementation path \
             cannot bind"
                .into(),
        );
    }
    verify_const_param_list(const_params(attr(op, "const_params"))?, false)
}

/// `preview$extern_impl` on the native SIMT path: fragment arguments only,
/// no results, no `config`, per-argument access matching a contiguous run,
/// consistent fragmaps and a participant count the synchronization accepts.
fn verify_extern_impl(op: &Operation) -> Result<(), String> {
    symbol(op, "sym_name")?;
    symbol(op, "implements")?;
    let func = function_type(op)?;
    if attr(op, "config").is_some() {
        return Err("extern_impl must not carry 'config'".into());
    }
    if !func.results.is_empty() {
        return Err(format!(
            "extern_impl must not declare results; outputs are represented as 'out' or 'inout' \
             fragment args, got {} result(s)",
            func.results.len()
        ));
    }
    let fragments: Vec<&FragmentType> = func
        .inputs
        .iter()
        .map(|ty| match ty {
            Type::Fragment(f) => Some(f),
            _ => None,
        })
        .collect::<Option<_>>()
        .filter(|f: &Vec<_>| !f.is_empty())
        .ok_or(
            "function_type must have at least one input and all inputs must be fragment types",
        )?;
    let accesses = argument_accesses(attr(op, "arg_attrs"), fragments.len())?;
    verify_access_run(&accesses)?;
    let params = const_params(attr(op, "const_params"))?;
    verify_const_param_list(params.iter().copied(), true)?;
    let declared = declared_variables(params);
    if let Some(args) = attr(op, "const_args") {
        match args {
            Attribute::Array(args) => {
                if let Some(i) = args
                    .iter()
                    .position(|a| !matches!(a, Attribute::SymExpr(_)))
                {
                    return Err(format!(
                        "constant argument {i} must be a symbolic expression attribute"
                    ));
                }
            }
            _ => return Err("const_args must be an array".into()),
        }
    }
    if let Some(arch) = attr(op, "arch") {
        match arch {
            Attribute::Array(items) => {
                for (i, item) in items.iter().enumerate() {
                    if !matches!(item, Attribute::String(s) if !s.is_empty()) {
                        return Err(format!("arch[{i}] must be a non-empty string"));
                    }
                }
            }
            _ => return Err("arch must be an array of strings".into()),
        }
    }
    if let Some(scratch) = attr(op, "scratch_smem") {
        match scratch {
            Attribute::ScratchRequest(scratch) => {
                verify_scratch(scratch)?;
                verify_scratch_variables(scratch, &declared)?;
            }
            _ => return Err("scratch_smem must be a scratch request".into()),
        }
    }
    let synchronization = match attr(op, "synchronization") {
        Some(Attribute::Synchronization(s)) => *s,
        None => Synchronization::None,
        Some(_) => return Err("synchronization must be a synchronization attribute".into()),
    };

    let mut shared_participants: Option<(i64, usize)> = None;
    let mut shared_invocations: Option<(i64, usize)> = None;
    let mut all_concrete = true;
    for (i, (fragment, access)) in fragments.iter().zip(&accesses).enumerate() {
        verify_fragment(fragment).map_err(|e| format!("fragment arg {i}: {e}"))?;
        verify_fragment_variables(fragment, &declared)
            .map_err(|e| format!("fragment arg {i}: {e}"))?;
        // This crate's narrowing: it emits a rank-0 mem_layout only for the
        // one case the SIMT ABI promotes to a by-value scalar. Upstream
        // accepts rank 0 on any fragment and passes an out/inout one as a
        // pointer.
        if fragment.mem_layout.rank() == 0
            && (*access != Access::In || fragment.memory_space != MemorySpace::Local)
        {
            return Err(format!(
                "fragment arg {i}: this crate only emits a rank-0 mem_layout (by-value scalar) \
                 for an 'in' argument in local memory; upstream would pass an out/inout rank-0 \
                 fragment as a pointer"
            ));
        }
        if !fragment.fragmap.is_concrete() || !fragment.mem_layout.is_concrete() {
            all_concrete = false;
            continue;
        }
        let shapes = &fragment.fragmap.shapes;
        let strides = &fragment.fragmap.strides;
        let (Some(inv), Some(part), Some(foot)) = (
            mode_size(&shapes[0]),
            mode_size(&shapes[1]),
            mode_size(&shapes[2]),
        ) else {
            return Err(format!(
                "fragment arg {i} fragmap modes must contain signed 64-bit shape leaves with \
                 representable products"
            ));
        };
        let elements = fragment.num_elements().unwrap_or(0);
        if foot != elements {
            return Err(format!(
                "fragment arg {i} fragmap footprint mode size ({foot}) must equal the fragment \
                 element count ({elements})"
            ));
        }
        if fragment.memory_space == MemorySpace::Smem
            && mode_has_zero_stride(&shapes[1], &strides[1]) != Some(true)
        {
            return Err(format!(
                "fragment arg {i} smem fragmap participant mode must have zero stride for \
                 every non-unit submode"
            ));
        }
        match shared_participants {
            None => shared_participants = Some((part, i)),
            Some((shared, reference)) if shared != part => {
                return Err(format!(
                    "fragment arg {i} fragmap participant mode size ({part}) must match arg \
                     {reference}'s ({shared}); all args of one invocation use the same participants"
                ))
            }
            _ => {}
        }
        match shared_invocations {
            None => shared_invocations = Some((inv, i)),
            Some((shared, reference)) if shared != inv => {
                return Err(format!(
                    "fragment arg {i} fragmap invocation mode size ({inv}) must match arg \
                     {reference}'s ({shared})"
                ))
            }
            _ => {}
        }
    }
    if all_concrete {
        if let Some((participants, _)) = shared_participants {
            verify_synchronization_participation(synchronization, participants)?;
        }
    }
    Ok(())
}

/// `preview$call`: a named callee, statically known constant arguments, and
/// tile or token operands and results. A call instantiates a declaration,
/// so its constant arguments may not forward a symbolic variable or pass an
/// empty tuple (`verifyCallConstArgIsStatic`); context builtins have a
/// bounded definition and stay allowed.
fn verify_preview_call(module: &Module, op: &Operation) -> Result<(), String> {
    symbol(op, "callee")?;
    match attr(op, "const_args") {
        None => {}
        Some(Attribute::Array(args)) => {
            let none = HashSet::new();
            for (i, arg) in args.iter().enumerate() {
                let Attribute::SymExpr(expr) = arg else {
                    return Err(format!(
                        "constant argument {i} must be a symbolic expression attribute"
                    ));
                };
                let empty = matches!(expr, SymExpr::Tuple(values) if values.is_empty());
                let variable = undeclared_variable(expr, &none);
                if empty {
                    return Err(format!(
                        "constant argument {i} must be statically known; an empty tuple is \
                         not supported"
                    ));
                }
                if let Some(name) = variable {
                    return Err(format!(
                        "constant argument {i} must be statically known; it references the \
                         symbolic variable '{name}'"
                    ));
                }
            }
        }
        Some(_) => return Err("const_args must be an array".into()),
    }
    let boundary = |ty: &Type| matches!(ty, Type::Tile(_) | Type::Token);
    if let Some(i) = op
        .operands
        .iter()
        .position(|v| !boundary(module.value_type(*v)))
    {
        return Err(format!("operand {i} must be a cuda_tile.tile or token"));
    }
    if let Some(i) = op.result_types.iter().position(|ty| !boundary(ty)) {
        return Err(format!("result {i} must be a cuda_tile.tile or token"));
    }
    Ok(())
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
        PreviewFunc | PreviewExternImpl | PreviewCall => {
            verify_preview_op(module, op).map_err(|e| fail(&e))?
        }
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
