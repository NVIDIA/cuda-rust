/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tile IR type system.
//!
//! Mirrors the CUDA Tile dialect types defined in `Types.td`.

use crate::bytecode::TypeTag;

/// Marker for a dimension or stride whose size is not statically known.
/// Matches MLIR's `ShapedType::kDynamic` (`i64::MIN`); printed as `?` in text.
pub const DYNAMIC: i64 = i64::MIN;

// ---------------------------------------------------------------------------
// Scalar types
// ---------------------------------------------------------------------------

/// Scalar (element) types — integers and floats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    I1,
    I4,
    I8,
    I16,
    I32,
    I64,
    F16,
    BF16,
    F32,
    TF32,
    F64,
    F8E4M3FN,
    F8E5M2,
    F8E8M0FNU,
    F4E2M1FN,
    F8E5M3FNU,
}

impl ScalarType {
    /// Minimum bytecode version that can represent this element type.
    pub const fn minimum_version(self) -> crate::bytecode::BytecodeVersion {
        crate::requirements::scalar_requirement(self).since
    }

    /// Byte width of this scalar type (I1 rounds up to 1 byte).
    pub fn byte_width(self) -> usize {
        match self {
            Self::I1
            | Self::I4
            | Self::I8
            | Self::F8E4M3FN
            | Self::F8E5M2
            | Self::F8E8M0FNU
            | Self::F4E2M1FN
            | Self::F8E5M3FNU => 1,
            Self::I16 | Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::F32 | Self::TF32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }

    pub fn type_tag(self) -> TypeTag {
        match self {
            Self::I1 => TypeTag::I1,
            Self::I4 => TypeTag::I4,
            Self::I8 => TypeTag::I8,
            Self::I16 => TypeTag::I16,
            Self::I32 => TypeTag::I32,
            Self::I64 => TypeTag::I64,
            Self::F16 => TypeTag::F16,
            Self::BF16 => TypeTag::BF16,
            Self::F32 => TypeTag::F32,
            Self::TF32 => TypeTag::TF32,
            Self::F64 => TypeTag::F64,
            Self::F8E4M3FN => TypeTag::F8E4M3FN,
            Self::F8E5M2 => TypeTag::F8E5M2,
            Self::F8E8M0FNU => TypeTag::F8E8M0FNU,
            Self::F4E2M1FN => TypeTag::F4E2M1FN,
            Self::F8E5M3FNU => TypeTag::F8E5M3FNU,
        }
    }

    pub fn is_integer(self) -> bool {
        matches!(
            self,
            Self::I1 | Self::I4 | Self::I8 | Self::I16 | Self::I32 | Self::I64
        )
    }

    pub fn is_float(self) -> bool {
        !self.is_integer()
    }
}

// ---------------------------------------------------------------------------
// Compound types
// ---------------------------------------------------------------------------

/// An element type that can appear inside a Tile (scalar or pointer).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TileElementType {
    Scalar(ScalarType),
    Pointer(Box<PointerType>),
}

/// Pointer to a scalar value in global device memory.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PointerType {
    pub pointee: ScalarType,
}

/// Public Tile IR 13.4 pointer classifications. Absence is represented by an
/// unqualified type; explicit `None` remains distinct from absence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PointerAttribute {
    None = 0,
}

/// A statically-shaped tile of elements.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TileType {
    pub shape: Vec<i64>,
    pub element_type: TileElementType,
}

/// A reference to a tensor in global memory with shape and strides.
/// Use [`DYNAMIC`] for dimensions/strides that are not statically known.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorViewType {
    pub element_type: ScalarType,
    pub shape: Vec<i64>,
    pub strides: Vec<i64>,
}

/// A view into a tensor where tiles are laid out in a grid pattern.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PartitionViewType {
    pub tile_shape: Vec<i32>,
    pub tensor_view: TensorViewType,
    pub dim_map: Vec<i32>,
    pub padding_value: Option<PaddingValue>,
}

/// A view that uses a 1D tensor of sparse indices along one tensor dimension.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GatherScatterViewType {
    pub tile_shape: Vec<i32>,
    pub tensor_view: TensorViewType,
    pub sparse_dim: i32,
    pub padding_value: Option<PaddingValue>,
}

/// A view that traverses a tensor with configurable per-dimension strides.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StridedViewType {
    pub tile_shape: Vec<i32>,
    pub traversal_strides: Vec<i32>,
    pub tensor_view: TensorViewType,
    pub dim_map: Vec<i32>,
    pub padding_value: Option<PaddingValue>,
}

/// Padding value for out-of-bounds accesses in a partition view.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PaddingValue {
    Zero = 0,
    NegZero = 1,
    Nan = 2,
    PosInf = 3,
    NegInf = 4,
}

/// Function signature type.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FuncType {
    pub inputs: Vec<Type>,
    pub results: Vec<Type>,
}

/// The unified type enum — every value in the IR has one of these types.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Type {
    Scalar(ScalarType),
    Pointer(PointerType),
    Tile(TileType),
    TensorView(TensorViewType),
    PartitionView(PartitionViewType),
    GatherScatterView(GatherScatterViewType),
    StridedView(StridedViewType),
    Func(FuncType),
    Token,
    /// Attach a classification to a pointer, pointer tile, or a view's base
    /// tensor. A wrapper preserves the existing public carrier structs.
    WithPointerAttribute(Box<Type>, PointerAttribute),
}

impl Type {
    pub fn with_pointer_attribute(self, attribute: PointerAttribute) -> Self {
        Self::WithPointerAttribute(
            Box::new(self.without_pointer_attribute().clone()),
            attribute,
        )
    }

    pub fn without_pointer_attribute(&self) -> &Type {
        match self {
            Self::WithPointerAttribute(base, _) => base.without_pointer_attribute(),
            _ => self,
        }
    }

    pub fn pointer_attribute(&self) -> Option<PointerAttribute> {
        match self {
            Self::WithPointerAttribute(_, attr) => Some(*attr),
            _ => None,
        }
    }
}

impl From<ScalarType> for Type {
    fn from(s: ScalarType) -> Self {
        Self::Scalar(s)
    }
}

impl From<PointerType> for Type {
    fn from(p: PointerType) -> Self {
        Self::Pointer(p)
    }
}

impl From<TileType> for Type {
    fn from(t: TileType) -> Self {
        Self::Tile(t)
    }
}

impl From<TensorViewType> for Type {
    fn from(t: TensorViewType) -> Self {
        Self::TensorView(t)
    }
}

impl From<PartitionViewType> for Type {
    fn from(p: PartitionViewType) -> Self {
        Self::PartitionView(p)
    }
}

impl From<GatherScatterViewType> for Type {
    fn from(g: GatherScatterViewType) -> Self {
        Self::GatherScatterView(g)
    }
}

impl From<StridedViewType> for Type {
    fn from(s: StridedViewType) -> Self {
        Self::StridedView(s)
    }
}

impl From<FuncType> for Type {
    fn from(f: FuncType) -> Self {
        Self::Func(f)
    }
}

// ---------------------------------------------------------------------------
// Type parser — equivalent of melior::ir::Type::parse(ctx, s)
// ---------------------------------------------------------------------------

impl Type {
    /// Parse a CUDA Tile MLIR type string into a `Type`.
    ///
    /// This is the tile-ir equivalent of `melior::ir::Type::parse(ctx, s)`.
    /// It handles the full grammar of CUDA Tile dialect types:
    ///
    /// - `!cuda_tile.tile<[shape x]elem>` where elem is scalar or `!cuda_tile.ptr<scalar>`
    /// - `!cuda_tile.tensor_view<[shape x]scalar, strides=[strides]>`
    /// - `!cuda_tile.partition_view<tile=(dims), [padding_value=X,] tensor_view<...>[, dim_map=[...]]>`
    /// - `!cuda_tile.token`
    /// - Bare scalar names (e.g. `f32`, `i32`)
    ///
    /// Returns `None` if the string doesn't match any known type.
    pub fn parse(s: &str) -> Option<Type> {
        let s = s.trim();
        if s.contains(", #cuda_tile.ptr_attr<none>") {
            return Self::parse(&s.replace(", #cuda_tile.ptr_attr<none>", ""))
                .map(|ty| ty.with_pointer_attribute(PointerAttribute::None));
        }
        // Accept both `!cuda_tile.token` and shorthand `token`.
        if s == "!cuda_tile.token" || s == "token" {
            return Some(Type::Token);
        }
        // Prefixed forms.
        if let Some(inner) = strip_prefix_suffix(s, "!cuda_tile.tile<", ">") {
            return parse_tile(inner);
        }
        if let Some(inner) = strip_prefix_suffix(s, "!cuda_tile.tensor_view<", ">") {
            return parse_tensor_view(inner);
        }
        if let Some(inner) = strip_prefix_suffix(s, "!cuda_tile.partition_view<", ">") {
            return parse_partition_view(inner);
        }
        if let Some(inner) = strip_prefix_suffix(s, "!cuda_tile.gather_scatter_view<", ">") {
            return parse_gather_scatter_view(inner);
        }
        if let Some(inner) = strip_prefix_suffix(s, "!cuda_tile.strided_view<", ">") {
            return parse_strided_view(inner);
        }
        if let Some(inner) = strip_prefix_suffix(s, "!cuda_tile.ptr<", ">") {
            let pointee = parse_scalar(inner)?;
            return Some(Type::Pointer(PointerType { pointee }));
        }
        // Shorthand forms (no `!cuda_tile.` prefix).
        if let Some(inner) = strip_prefix_suffix(s, "tile<", ">") {
            return parse_tile(inner);
        }
        if let Some(inner) = strip_prefix_suffix(s, "tensor_view<", ">") {
            return parse_tensor_view(inner);
        }
        if let Some(inner) = strip_prefix_suffix(s, "partition_view<", ">") {
            return parse_partition_view(inner);
        }
        if let Some(inner) = strip_prefix_suffix(s, "gather_scatter_view<", ">") {
            return parse_gather_scatter_view(inner);
        }
        if let Some(inner) = strip_prefix_suffix(s, "strided_view<", ">") {
            return parse_strided_view(inner);
        }
        if let Some(inner) = strip_prefix_suffix(s, "ptr<", ">") {
            let pointee = parse_scalar(inner)?;
            return Some(Type::Pointer(PointerType { pointee }));
        }
        // Bare scalar name.
        parse_scalar(s).map(Type::Scalar)
    }
}

impl ScalarType {
    /// Parse a scalar type name string.
    pub fn parse(s: &str) -> Option<ScalarType> {
        parse_scalar(s)
    }
}

/// Strip a known prefix and the matching closing `>`, handling nested `<>`.
fn strip_prefix_suffix<'a>(s: &'a str, prefix: &str, _suffix: &str) -> Option<&'a str> {
    if !s.starts_with(prefix) {
        return None;
    }
    let after_prefix = &s[prefix.len()..];
    // Find the matching closing '>' by counting nesting depth.
    let mut depth = 1;
    for (i, c) in after_prefix.char_indices() {
        match c {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&after_prefix[..i]);
                }
            }
            _ => {}
        }
    }
    // No matching close — try without nesting (just strip last char if it's '>').
    after_prefix.strip_suffix('>')
}

fn parse_scalar(s: &str) -> Option<ScalarType> {
    match s.trim() {
        "i1" => Some(ScalarType::I1),
        "i4" => Some(ScalarType::I4),
        "i8" => Some(ScalarType::I8),
        "i16" => Some(ScalarType::I16),
        "i32" => Some(ScalarType::I32),
        "i64" => Some(ScalarType::I64),
        "f16" => Some(ScalarType::F16),
        "bf16" => Some(ScalarType::BF16),
        "f32" => Some(ScalarType::F32),
        "tf32" => Some(ScalarType::TF32),
        "f64" => Some(ScalarType::F64),
        "f8E4M3FN" | "f8e4m3fn" => Some(ScalarType::F8E4M3FN),
        "f8E5M2" | "f8e5m2" => Some(ScalarType::F8E5M2),
        "f8E8M0FNU" | "f8e8m0fnu" => Some(ScalarType::F8E8M0FNU),
        "f4E2M1FN" | "f4e2m1fn" => Some(ScalarType::F4E2M1FN),
        "f8E5M3FNU" | "f8e5m3fnu" => Some(ScalarType::F8E5M3FNU),
        // Rust-facing names used by the compiler.
        "bool" => Some(ScalarType::I1),
        _ => None,
    }
}

/// Parse a dim string: integer or `?` for DYNAMIC.
fn parse_dim(s: &str) -> i64 {
    let s = s.trim();
    if s == "?" {
        DYNAMIC
    } else {
        s.parse::<i64>().unwrap_or(DYNAMIC)
    }
}

/// Parse `[shape x]elem` where shape is `dim[xdim]*` and elem is a scalar or `!cuda_tile.ptr<scalar>`.
fn parse_tile(inner: &str) -> Option<Type> {
    // Check for pointer element: `!cuda_tile.ptr<scalar>` or `ptr<scalar>`
    if let Some(ptr_start) = inner.find("ptr<") {
        let before = inner[..ptr_start].trim().trim_end_matches("!cuda_tile.");
        let shape: Vec<i64> = if before.is_empty() {
            vec![]
        } else {
            before
                .trim_end_matches('x')
                .split('x')
                .map(parse_dim)
                .collect()
        };
        let ptr_inner_start = ptr_start + "ptr<".len();
        let ptr_inner_end = inner[ptr_inner_start..].find('>')?;
        let pointee = parse_scalar(&inner[ptr_inner_start..ptr_inner_start + ptr_inner_end])?;
        return Some(Type::Tile(TileType {
            shape,
            element_type: TileElementType::Pointer(Box::new(PointerType { pointee })),
        }));
    }

    // Split on 'x' to get shape dims and trailing element type.
    // e.g. "128xf32" → ["128", "f32"], "f32" → ["f32"], "4x128xf32" → ["4", "128", "f32"]
    let parts: Vec<&str> = inner.split('x').collect();
    if parts.is_empty() {
        return None;
    }
    let elem_str = parts.last()?;
    let scalar = parse_scalar(elem_str)?;
    let shape: Vec<i64> = parts[..parts.len() - 1]
        .iter()
        .map(|p| parse_dim(p))
        .collect();
    Some(Type::Tile(TileType {
        shape,
        element_type: TileElementType::Scalar(scalar),
    }))
}

/// Parse `[shape x]scalar, strides=[strides]`.
fn parse_tensor_view(inner: &str) -> Option<Type> {
    // Split at ", strides=" to separate shape+elem from strides.
    let (shape_elem, strides_part) = if let Some(pos) = inner.find(", strides=") {
        (&inner[..pos], Some(&inner[pos + ", strides=".len()..]))
    } else if let Some(pos) = inner.find(",strides=") {
        (&inner[..pos], Some(&inner[pos + ",strides=".len()..]))
    } else {
        (inner, None)
    };

    let parts: Vec<&str> = shape_elem.split('x').collect();
    if parts.is_empty() {
        return None;
    }
    let elem_str = parts.last()?;
    let scalar = parse_scalar(elem_str)?;
    let shape: Vec<i64> = parts[..parts.len() - 1]
        .iter()
        .map(|p| parse_dim(p))
        .collect();

    let strides = if let Some(sp) = strides_part {
        let sp = sp.trim_start_matches('[').trim_end_matches(']');
        sp.split(',').map(parse_dim).collect()
    } else {
        vec![DYNAMIC; shape.len()]
    };

    Some(Type::TensorView(TensorViewType {
        element_type: scalar,
        shape,
        strides,
    }))
}

/// Parse `tile=(dims), [padding_value=X,] tensor_view<...>[, dim_map=[...]]`.
fn parse_partition_view(inner: &str) -> Option<Type> {
    // Extract tile shape from "tile=(d1xd2)" or "tile=(d1, d2)" or "tile=(d1)".
    let tile_start = inner.find("tile=(")?;
    let dims_start = tile_start + "tile=(".len();
    let dims_end = inner[dims_start..].find(')')? + dims_start;
    let dims_str = &inner[dims_start..dims_end];
    let tile_shape: Vec<i32> = if dims_str.contains('x') {
        dims_str
            .split('x')
            .map(|s| s.trim().parse::<i32>().unwrap_or(1))
            .collect()
    } else {
        dims_str
            .split(',')
            .map(|s| s.trim().parse::<i32>().unwrap_or(1))
            .collect()
    };

    // Extract padding_value if present.
    let padding_value = if inner.contains("padding_value") {
        if inner.contains("zero") && !inner.contains("neg_zero") {
            Some(PaddingValue::Zero)
        } else if inner.contains("neg_zero") {
            Some(PaddingValue::NegZero)
        } else if inner.contains("nan") {
            Some(PaddingValue::Nan)
        } else if inner.contains("pos_inf") {
            Some(PaddingValue::PosInf)
        } else if inner.contains("neg_inf") {
            Some(PaddingValue::NegInf)
        } else {
            None
        }
    } else {
        None
    };

    // Extract the tensor_view — may be "tensor_view=!cuda_tile.tensor_view<...>"
    // or just "tensor_view<...>" or "!cuda_tile.tensor_view<...>".
    let tv_search = inner;
    let tv_prefix_start = tv_search
        .find("tensor_view=!cuda_tile.tensor_view<")
        .map(|p| (p, "tensor_view=!cuda_tile.tensor_view<"))
        .or_else(|| {
            tv_search
                .find("!cuda_tile.tensor_view<")
                .map(|p| (p, "!cuda_tile.tensor_view<"))
        })
        .or_else(|| {
            // "tensor_view<" without the "tensor_view=" or "!cuda_tile." prefix
            // but NOT matching the "tile=" prefix that also appears
            let remaining = &tv_search[dims_end + 1..]; // after the tile=(...) part
            remaining
                .find("tensor_view<")
                .map(|p| (p + dims_end + 1, "tensor_view<"))
        })?;
    let (tv_pos, tv_prefix) = tv_prefix_start;
    let tv_inner_start = tv_pos + tv_prefix.len();

    // Find matching closing '>' for the tensor_view.
    let mut depth = 1;
    let mut tv_inner_end = tv_inner_start;
    for (i, c) in inner[tv_inner_start..].char_indices() {
        match c {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    tv_inner_end = tv_inner_start + i;
                    break;
                }
            }
            _ => {}
        }
    }
    let tv_inner = &inner[tv_inner_start..tv_inner_end];
    let tv_type = parse_tensor_view(tv_inner)?;
    let Type::TensorView(tv) = tv_type else {
        return None;
    };

    // Extract dim_map if present.
    let dim_map = if let Some(dm_start) = inner.find("dim_map=[") {
        let dm_inner_start = dm_start + "dim_map=[".len();
        let dm_end = inner[dm_inner_start..].find(']')? + dm_inner_start;
        inner[dm_inner_start..dm_end]
            .split(',')
            .map(|s| s.trim().parse::<i32>().unwrap_or(0))
            .collect()
    } else {
        (0..tile_shape.len() as i32).collect()
    };

    Some(Type::PartitionView(PartitionViewType {
        tile_shape,
        tensor_view: tv,
        dim_map,
        padding_value,
    }))
}

fn parse_view_tile_shape(inner: &str) -> Option<(Vec<i32>, usize)> {
    let tile_start = inner.find("tile=(")?;
    let dims_start = tile_start + "tile=(".len();
    let dims_end = inner[dims_start..].find(')')? + dims_start;
    let dims_str = &inner[dims_start..dims_end];
    let tile_shape: Vec<i32> = if dims_str.contains('x') {
        dims_str
            .split('x')
            .map(|s| s.trim().parse::<i32>().unwrap_or(1))
            .collect()
    } else {
        dims_str
            .split(',')
            .map(|s| s.trim().parse::<i32>().unwrap_or(1))
            .collect()
    };
    Some((tile_shape, dims_end))
}

fn parse_padding_value(inner: &str) -> Option<PaddingValue> {
    if inner.contains("padding_value") {
        if inner.contains("zero") && !inner.contains("neg_zero") {
            Some(PaddingValue::Zero)
        } else if inner.contains("neg_zero") {
            Some(PaddingValue::NegZero)
        } else if inner.contains("nan") {
            Some(PaddingValue::Nan)
        } else if inner.contains("pos_inf") {
            Some(PaddingValue::PosInf)
        } else if inner.contains("neg_inf") {
            Some(PaddingValue::NegInf)
        } else {
            None
        }
    } else {
        None
    }
}

fn parse_nested_tensor_view(inner: &str, after_pos: usize) -> Option<TensorViewType> {
    let tv_search = inner;
    let tv_prefix_start = tv_search
        .find("tensor_view=!cuda_tile.tensor_view<")
        .map(|p| (p, "tensor_view=!cuda_tile.tensor_view<"))
        .or_else(|| {
            tv_search
                .find("!cuda_tile.tensor_view<")
                .map(|p| (p, "!cuda_tile.tensor_view<"))
        })
        .or_else(|| {
            let remaining = &tv_search[after_pos..];
            remaining
                .find("tensor_view<")
                .map(|p| (p + after_pos, "tensor_view<"))
        })?;
    let (tv_pos, tv_prefix) = tv_prefix_start;
    let tv_inner_start = tv_pos + tv_prefix.len();

    let mut depth = 1;
    let mut tv_inner_end = tv_inner_start;
    for (i, c) in inner[tv_inner_start..].char_indices() {
        match c {
            '<' => depth += 1,
            '>' => {
                depth -= 1;
                if depth == 0 {
                    tv_inner_end = tv_inner_start + i;
                    break;
                }
            }
            _ => {}
        }
    }
    let tv_inner = &inner[tv_inner_start..tv_inner_end];
    let tv_type = parse_tensor_view(tv_inner)?;
    let Type::TensorView(tv) = tv_type else {
        return None;
    };
    Some(tv)
}

fn parse_i32_list_after(inner: &str, prefix: &str) -> Option<Vec<i32>> {
    let start = inner.find(prefix)? + prefix.len();
    let end = inner[start..].find(']')? + start;
    Some(
        inner[start..end]
            .split(',')
            .map(|s| s.trim().parse::<i32>().unwrap_or(0))
            .collect(),
    )
}

fn parse_gather_scatter_view(inner: &str) -> Option<Type> {
    let (tile_shape, dims_end) = parse_view_tile_shape(inner)?;
    let padding_value = parse_padding_value(inner);
    let tensor_view = parse_nested_tensor_view(inner, dims_end + 1)?;
    let sparse_dim = if let Some(start) = inner.find("sparse_dim=") {
        let value_start = start + "sparse_dim=".len();
        let value = inner[value_start..]
            .split([',', '>'])
            .next()
            .unwrap_or("0")
            .trim();
        value.parse::<i32>().unwrap_or(0)
    } else {
        0
    };

    Some(Type::GatherScatterView(GatherScatterViewType {
        tile_shape,
        tensor_view,
        sparse_dim,
        padding_value,
    }))
}

fn parse_strided_view(inner: &str) -> Option<Type> {
    let (tile_shape, dims_end) = parse_view_tile_shape(inner)?;
    let padding_value = parse_padding_value(inner);
    let traversal_strides = parse_i32_list_after(inner, "traversal_strides=[")
        .unwrap_or_else(|| vec![1; tile_shape.len()]);
    let tensor_view = parse_nested_tensor_view(inner, dims_end + 1)?;
    let dim_map = parse_i32_list_after(inner, "dim_map=[")
        .unwrap_or_else(|| (0..tile_shape.len() as i32).collect());

    Some(Type::StridedView(StridedViewType {
        tile_shape,
        traversal_strides,
        tensor_view,
        dim_map,
        padding_value,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Scalars --

    #[test]
    fn parse_scalar_f32() {
        assert_eq!(Type::parse("f32"), Some(Type::Scalar(ScalarType::F32)));
    }

    #[test]
    fn parse_scalar_i32() {
        assert_eq!(Type::parse("i32"), Some(Type::Scalar(ScalarType::I32)));
    }

    #[test]
    fn parse_scalar_bf16() {
        assert_eq!(Type::parse("bf16"), Some(Type::Scalar(ScalarType::BF16)));
    }

    #[test]
    fn parse_cuda_13_3_scalars() {
        assert_eq!(Type::parse("i4"), Some(Type::Scalar(ScalarType::I4)));
        assert_eq!(
            Type::parse("f8E8M0FNU"),
            Some(Type::Scalar(ScalarType::F8E8M0FNU))
        );
        assert_eq!(
            Type::parse("f4E2M1FN"),
            Some(Type::Scalar(ScalarType::F4E2M1FN))
        );
        assert_eq!(
            Type::parse("!cuda_tile.tile<64xf4E2M1FN>"),
            Some(Type::Tile(TileType {
                shape: vec![64],
                element_type: TileElementType::Scalar(ScalarType::F4E2M1FN),
            }))
        );
    }

    #[test]
    fn parse_unknown_returns_none() {
        assert_eq!(Type::parse("not_a_type"), None);
    }

    // -- Token --

    #[test]
    fn parse_token() {
        assert_eq!(Type::parse("!cuda_tile.token"), Some(Type::Token));
    }

    // -- Tile (scalar element) --

    #[test]
    fn parse_tile_scalar() {
        assert_eq!(
            Type::parse("!cuda_tile.tile<f32>"),
            Some(Type::Tile(TileType {
                shape: vec![],
                element_type: TileElementType::Scalar(ScalarType::F32),
            }))
        );
    }

    #[test]
    fn parse_tile_1d() {
        assert_eq!(
            Type::parse("!cuda_tile.tile<128xf32>"),
            Some(Type::Tile(TileType {
                shape: vec![128],
                element_type: TileElementType::Scalar(ScalarType::F32),
            }))
        );
    }

    #[test]
    fn parse_tile_2d() {
        assert_eq!(
            Type::parse("!cuda_tile.tile<16x8xf32>"),
            Some(Type::Tile(TileType {
                shape: vec![16, 8],
                element_type: TileElementType::Scalar(ScalarType::F32),
            }))
        );
    }

    #[test]
    fn parse_tile_i32() {
        assert_eq!(
            Type::parse("!cuda_tile.tile<i32>"),
            Some(Type::Tile(TileType {
                shape: vec![],
                element_type: TileElementType::Scalar(ScalarType::I32),
            }))
        );
    }

    // -- Tile (pointer element) --

    #[test]
    fn parse_tile_ptr() {
        assert_eq!(
            Type::parse("!cuda_tile.tile<!cuda_tile.ptr<f32>>"),
            Some(Type::Tile(TileType {
                shape: vec![],
                element_type: TileElementType::Pointer(Box::new(PointerType {
                    pointee: ScalarType::F32
                })),
            }))
        );
    }

    #[test]
    fn parse_tile_ptr_i64() {
        assert_eq!(
            Type::parse("!cuda_tile.tile<!cuda_tile.ptr<i64>>"),
            Some(Type::Tile(TileType {
                shape: vec![],
                element_type: TileElementType::Pointer(Box::new(PointerType {
                    pointee: ScalarType::I64
                })),
            }))
        );
    }

    // -- TensorView --

    #[test]
    fn parse_tensor_view_static() {
        assert_eq!(
            Type::parse("!cuda_tile.tensor_view<128xf32, strides=[1]>"),
            Some(Type::TensorView(TensorViewType {
                element_type: ScalarType::F32,
                shape: vec![128],
                strides: vec![1],
            }))
        );
    }

    #[test]
    fn parse_tensor_view_dynamic() {
        assert_eq!(
            Type::parse("!cuda_tile.tensor_view<?xf32, strides=[?]>"),
            Some(Type::TensorView(TensorViewType {
                element_type: ScalarType::F32,
                shape: vec![DYNAMIC],
                strides: vec![DYNAMIC],
            }))
        );
    }

    #[test]
    fn parse_tensor_view_2d() {
        assert_eq!(
            Type::parse("!cuda_tile.tensor_view<?x8192xf32, strides=[?,1]>"),
            Some(Type::TensorView(TensorViewType {
                element_type: ScalarType::F32,
                shape: vec![DYNAMIC, 8192],
                strides: vec![DYNAMIC, 1],
            }))
        );
    }

    // -- PartitionView --

    #[test]
    fn parse_partition_view_1d_melior() {
        // Melior format: tensor_view<...> without = prefix.
        assert_eq!(
            Type::parse("!cuda_tile.partition_view<tile=(128), tensor_view<128xf32, strides=[1]>>"),
            Some(Type::PartitionView(PartitionViewType {
                tile_shape: vec![128],
                tensor_view: TensorViewType {
                    element_type: ScalarType::F32,
                    shape: vec![128],
                    strides: vec![1],
                },
                dim_map: vec![0],
                padding_value: None,
            }))
        );
    }

    #[test]
    fn parse_partition_view_1d_formatted() {
        // tile-ir formatter format: tensor_view=!cuda_tile.tensor_view<...>.
        assert_eq!(
            Type::parse("!cuda_tile.partition_view<tile=(128), tensor_view=!cuda_tile.tensor_view<128xf32, strides=[1]>>"),
            Some(Type::PartitionView(PartitionViewType {
                tile_shape: vec![128],
                tensor_view: TensorViewType {
                    element_type: ScalarType::F32,
                    shape: vec![128],
                    strides: vec![1],
                },
                dim_map: vec![0],
                padding_value: None,
            }))
        );
    }

    #[test]
    fn parse_partition_view_2d() {
        assert_eq!(
            Type::parse(
                "!cuda_tile.partition_view<tile=(16x8), tensor_view<?x8192xf32, strides=[?,1]>>"
            ),
            Some(Type::PartitionView(PartitionViewType {
                tile_shape: vec![16, 8],
                tensor_view: TensorViewType {
                    element_type: ScalarType::F32,
                    shape: vec![DYNAMIC, 8192],
                    strides: vec![DYNAMIC, 1],
                },
                dim_map: vec![0, 1],
                padding_value: None,
            }))
        );
    }

    #[test]
    fn parse_partition_view_dynamic() {
        assert_eq!(
            Type::parse("!cuda_tile.partition_view<tile=(16), tensor_view<?xf32, strides=[?]>>"),
            Some(Type::PartitionView(PartitionViewType {
                tile_shape: vec![16],
                tensor_view: TensorViewType {
                    element_type: ScalarType::F32,
                    shape: vec![DYNAMIC],
                    strides: vec![DYNAMIC],
                },
                dim_map: vec![0],
                padding_value: None,
            }))
        );
    }

    #[test]
    fn parse_partition_view_with_padding() {
        let ty = Type::parse(
            "!cuda_tile.partition_view<tile=(8), padding_value = zero, tensor_view<128xf32, strides=[1]>>"
        );
        let Some(Type::PartitionView(pv)) = ty else {
            panic!("expected PartitionView");
        };
        assert_eq!(pv.tile_shape, vec![8]);
        assert_eq!(pv.padding_value, Some(PaddingValue::Zero));
    }

    // -- Round-trip: format → parse --
    //
    // These tests construct every Type variant with representative field
    // values, format it via `format_type`, parse the string back, and
    // assert equality. Any field that the formatter silently drops will
    // be caught here.

    /// Helper: format then parse, asserting round-trip equality.
    fn assert_roundtrip(ty: Type) {
        let formatted = crate::ir::fmt::format_type(&ty);
        assert_eq!(
            Type::parse(&formatted),
            Some(ty.clone()),
            "roundtrip failed for {ty:?}\n  formatted: {formatted}",
        );
    }

    // ---- Scalar: every ScalarType variant ----

    #[test]
    fn roundtrip_scalar_all_variants() {
        let scalars = [
            ScalarType::I1,
            ScalarType::I8,
            ScalarType::I16,
            ScalarType::I32,
            ScalarType::I64,
            ScalarType::F16,
            ScalarType::BF16,
            ScalarType::F32,
            ScalarType::TF32,
            ScalarType::F64,
            ScalarType::F8E4M3FN,
            ScalarType::F8E5M2,
        ];
        for s in scalars {
            assert_roundtrip(Type::Scalar(s));
        }
    }

    // ---- Token ----

    #[test]
    fn roundtrip_token() {
        assert_roundtrip(Type::Token);
    }

    // ---- Pointer ----

    #[test]
    fn roundtrip_pointer_all_pointee_types() {
        let pointees = [
            ScalarType::F32,
            ScalarType::I64,
            ScalarType::BF16,
            ScalarType::F8E4M3FN,
        ];
        for p in pointees {
            assert_roundtrip(Type::Pointer(PointerType { pointee: p }));
        }
    }

    // ---- Tile (scalar element) ----

    #[test]
    fn roundtrip_tile_scalar_empty_shape() {
        assert_roundtrip(Type::Tile(TileType {
            shape: vec![],
            element_type: TileElementType::Scalar(ScalarType::F32),
        }));
    }

    #[test]
    fn roundtrip_tile_scalar_1d() {
        assert_roundtrip(Type::Tile(TileType {
            shape: vec![128],
            element_type: TileElementType::Scalar(ScalarType::I32),
        }));
    }

    #[test]
    fn roundtrip_tile_scalar_2d() {
        assert_roundtrip(Type::Tile(TileType {
            shape: vec![16, 8],
            element_type: TileElementType::Scalar(ScalarType::BF16),
        }));
    }

    #[test]
    fn roundtrip_tile_scalar_3d() {
        assert_roundtrip(Type::Tile(TileType {
            shape: vec![4, 16, 8],
            element_type: TileElementType::Scalar(ScalarType::F64),
        }));
    }

    #[test]
    fn roundtrip_tile_all_scalar_elements() {
        let scalars = [
            ScalarType::I1,
            ScalarType::I8,
            ScalarType::I16,
            ScalarType::I32,
            ScalarType::I64,
            ScalarType::F16,
            ScalarType::BF16,
            ScalarType::F32,
            ScalarType::TF32,
            ScalarType::F64,
            ScalarType::F8E4M3FN,
            ScalarType::F8E5M2,
        ];
        for s in scalars {
            assert_roundtrip(Type::Tile(TileType {
                shape: vec![64],
                element_type: TileElementType::Scalar(s),
            }));
        }
    }

    // ---- Tile (pointer element) ----

    #[test]
    fn roundtrip_tile_ptr_empty_shape() {
        assert_roundtrip(Type::Tile(TileType {
            shape: vec![],
            element_type: TileElementType::Pointer(Box::new(PointerType {
                pointee: ScalarType::F32,
            })),
        }));
    }

    #[test]
    fn roundtrip_tile_ptr_1d() {
        assert_roundtrip(Type::Tile(TileType {
            shape: vec![32],
            element_type: TileElementType::Pointer(Box::new(PointerType {
                pointee: ScalarType::I64,
            })),
        }));
    }

    #[test]
    fn roundtrip_tile_ptr_2d() {
        assert_roundtrip(Type::Tile(TileType {
            shape: vec![4, 8],
            element_type: TileElementType::Pointer(Box::new(PointerType {
                pointee: ScalarType::BF16,
            })),
        }));
    }

    // ---- TensorView ----

    #[test]
    fn roundtrip_tensor_view_1d_static() {
        assert_roundtrip(Type::TensorView(TensorViewType {
            element_type: ScalarType::F32,
            shape: vec![128],
            strides: vec![1],
        }));
    }

    #[test]
    fn roundtrip_tensor_view_1d_dynamic() {
        assert_roundtrip(Type::TensorView(TensorViewType {
            element_type: ScalarType::F32,
            shape: vec![DYNAMIC],
            strides: vec![DYNAMIC],
        }));
    }

    #[test]
    fn roundtrip_tensor_view_2d_mixed() {
        assert_roundtrip(Type::TensorView(TensorViewType {
            element_type: ScalarType::F32,
            shape: vec![DYNAMIC, 8192],
            strides: vec![DYNAMIC, 1],
        }));
    }

    #[test]
    fn roundtrip_tensor_view_2d_all_static() {
        assert_roundtrip(Type::TensorView(TensorViewType {
            element_type: ScalarType::I32,
            shape: vec![64, 128],
            strides: vec![128, 1],
        }));
    }

    #[test]
    fn roundtrip_tensor_view_2d_all_dynamic() {
        assert_roundtrip(Type::TensorView(TensorViewType {
            element_type: ScalarType::BF16,
            shape: vec![DYNAMIC, DYNAMIC],
            strides: vec![DYNAMIC, DYNAMIC],
        }));
    }

    #[test]
    fn roundtrip_tensor_view_3d() {
        assert_roundtrip(Type::TensorView(TensorViewType {
            element_type: ScalarType::F16,
            shape: vec![DYNAMIC, 32, 64],
            strides: vec![DYNAMIC, 64, 1],
        }));
    }

    #[test]
    fn roundtrip_tensor_view_all_element_types() {
        let scalars = [
            ScalarType::I1,
            ScalarType::I8,
            ScalarType::I16,
            ScalarType::I32,
            ScalarType::I64,
            ScalarType::F16,
            ScalarType::BF16,
            ScalarType::F32,
            ScalarType::TF32,
            ScalarType::F64,
            ScalarType::F8E4M3FN,
            ScalarType::F8E5M2,
        ];
        for s in scalars {
            assert_roundtrip(Type::TensorView(TensorViewType {
                element_type: s,
                shape: vec![DYNAMIC],
                strides: vec![1],
            }));
        }
    }

    // ---- PartitionView: no padding ----

    #[test]
    fn roundtrip_partition_view_1d_no_padding() {
        assert_roundtrip(Type::PartitionView(PartitionViewType {
            tile_shape: vec![128],
            tensor_view: TensorViewType {
                element_type: ScalarType::F32,
                shape: vec![128],
                strides: vec![1],
            },
            dim_map: vec![0],
            padding_value: None,
        }));
    }

    #[test]
    fn roundtrip_partition_view_2d_no_padding() {
        assert_roundtrip(Type::PartitionView(PartitionViewType {
            tile_shape: vec![16, 8],
            tensor_view: TensorViewType {
                element_type: ScalarType::F32,
                shape: vec![DYNAMIC, 8192],
                strides: vec![DYNAMIC, 1],
            },
            dim_map: vec![0, 1],
            padding_value: None,
        }));
    }

    // ---- PartitionView: all PaddingValue variants ----

    #[test]
    fn roundtrip_partition_view_all_padding_variants() {
        let paddings = [
            PaddingValue::Zero,
            PaddingValue::NegZero,
            PaddingValue::Nan,
            PaddingValue::PosInf,
            PaddingValue::NegInf,
        ];
        for pv in paddings {
            assert_roundtrip(Type::PartitionView(PartitionViewType {
                tile_shape: vec![64],
                tensor_view: TensorViewType {
                    element_type: ScalarType::F32,
                    shape: vec![DYNAMIC],
                    strides: vec![1],
                },
                dim_map: vec![0],
                padding_value: Some(pv),
            }));
        }
    }

    // ---- PartitionView: dim_map ----

    #[test]
    fn roundtrip_partition_view_identity_dim_map() {
        assert_roundtrip(Type::PartitionView(PartitionViewType {
            tile_shape: vec![16, 8],
            tensor_view: TensorViewType {
                element_type: ScalarType::F32,
                shape: vec![DYNAMIC, 8192],
                strides: vec![DYNAMIC, 1],
            },
            dim_map: vec![0, 1],
            padding_value: Some(PaddingValue::Zero),
        }));
    }

    #[test]
    fn roundtrip_partition_view_non_identity_dim_map() {
        // Swapped dim_map — this MUST survive the roundtrip.
        assert_roundtrip(Type::PartitionView(PartitionViewType {
            tile_shape: vec![16, 8],
            tensor_view: TensorViewType {
                element_type: ScalarType::F32,
                shape: vec![DYNAMIC, 8192],
                strides: vec![DYNAMIC, 1],
            },
            dim_map: vec![1, 0],
            padding_value: None,
        }));
    }

    // ---- PartitionView: all element types ----

    #[test]
    fn roundtrip_partition_view_element_types() {
        for s in [ScalarType::BF16, ScalarType::I32, ScalarType::F8E4M3FN] {
            assert_roundtrip(Type::PartitionView(PartitionViewType {
                tile_shape: vec![64],
                tensor_view: TensorViewType {
                    element_type: s,
                    shape: vec![DYNAMIC],
                    strides: vec![1],
                },
                dim_map: vec![0],
                padding_value: Some(PaddingValue::Nan),
            }));
        }
    }

    // ---- Func (format only, parser does not handle Func) ----

    #[test]
    fn format_func_type_does_not_panic() {
        let ty = Type::Func(FuncType {
            inputs: vec![
                Type::Scalar(ScalarType::I32),
                Type::TensorView(TensorViewType {
                    element_type: ScalarType::F32,
                    shape: vec![DYNAMIC],
                    strides: vec![1],
                }),
            ],
            results: vec![Type::Scalar(ScalarType::I32)],
        });
        let formatted = crate::ir::fmt::format_type(&ty);
        // Func types don't roundtrip through Type::parse (parser does not
        // handle the `(inputs) -> (results)` syntax), but formatting must
        // not panic and must include all pieces.
        assert!(formatted.contains("i32"));
        assert!(formatted.contains("tensor_view"));
        assert!(formatted.contains("->"));
    }
}
