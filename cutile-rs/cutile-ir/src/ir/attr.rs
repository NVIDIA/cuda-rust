/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Tile IR attribute types.
//!
//! Mirrors the CUDA Tile dialect attributes defined in `AttrDefs.td`.

use super::types::{ScalarType, Type};

// ---------------------------------------------------------------------------
// Enum attributes
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Signedness {
    Unsigned = 0,
    Signed = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum IntegerOverflow {
    None = 0,
    NoSignedWrap = 1,
    NoUnsignedWrap = 2,
    NoWrap = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum RoundingMode {
    NearestEven = 0,
    Zero = 1,
    NegativeInf = 2,
    PositiveInf = 3,
    Approx = 4,
    Full = 5,
    NearestIntToZero = 6,
    NearestAway = 7,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ComparisonOrdering {
    Unordered = 0,
    Ordered = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ComparisonPredicate {
    Equal = 0,
    NotEqual = 1,
    LessThan = 2,
    LessThanOrEqual = 3,
    GreaterThan = 4,
    GreaterThanOrEqual = 5,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AtomicRMWMode {
    And = 0,
    Or = 1,
    Xor = 2,
    Add = 3,
    AddF = 4,
    Max = 5,
    Min = 6,
    UMax = 7,
    UMin = 8,
    Xchg = 9,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MemoryScope {
    TileBlock = 0,
    Device = 1,
    System = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MemoryOrdering {
    Weak = 0,
    Relaxed = 1,
    Acquire = 2,
    Release = 3,
    AcqRel = 4,
}

// ---------------------------------------------------------------------------
// Structured attributes
// ---------------------------------------------------------------------------

/// `div_by` assumption predicate.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct DivBy {
    pub divisor: u64,
    pub every: Option<i64>,
    pub along: Option<i64>,
}

/// `same_elements` assumption predicate.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SameElements {
    pub values: Vec<i64>,
}

/// `bounded` assumption predicate.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Bounded {
    pub lb: Option<i64>,
    pub ub: Option<i64>,
}

/// Per-architecture optimization hints dictionary.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizationHints {
    pub entries: Vec<(String, Vec<(String, Attribute)>)>,
}

// ---------------------------------------------------------------------------
// Public-preview SIMT interop attributes (AttrDefs.td, feature `simt`)
// ---------------------------------------------------------------------------

/// Name of the per-argument access dictionary entry on `preview$func` and
/// `preview$extern_impl` signatures.
pub const ACCESS_ATTR: &str = "cuda_tile.preview$access";

/// Name of the formatter-only op attribute holding the argument names of a
/// bodiless `preview$func` or `preview$extern_impl`, as an array of strings.
///
/// Unlike [`ACCESS_ATTR`], this is a convention of this crate, not an
/// upstream attribute: a declaration has no block arguments to carry names,
/// and the upstream printer writes `%arg<i>` for every one. The names let
/// `to_mlir_text` reproduce hand-written fixtures. The attribute has no
/// bytecode encoding: the writer skips it, and neither the key nor the names
/// reach the string table, so the bytes of a library do not depend on what
/// its arguments were called. A missing entry prints as `%argN`.
pub const ARG_NAMES_ATTR: &str = "arg_names";

/// How a callee uses a tile operand (`#cuda_tile.preview$access<...>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Access {
    In = 0,
    /// Written by the callee; the caller supplies a destination tile.
    Out = 1,
    /// Read and written in place.
    InOut = 2,
}

impl Access {
    /// The keyword inside `#cuda_tile.preview$access<...>`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::In => "in",
            Self::Out => "out",
            Self::InOut => "inout",
        }
    }

    /// The access mode a bytecode enum value decodes to.
    pub const fn from_u64(value: u64) -> Option<Self> {
        match value {
            0 => Some(Self::In),
            1 => Some(Self::Out),
            2 => Some(Self::InOut),
            _ => None,
        }
    }
}

/// Memory space of a fragment buffer (`local` or `smem`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum MemorySpace {
    /// A per-invocation buffer in the thread's own memory, reached through a generic pointer.
    Local = 0,
    /// A CTA-wide shared-memory buffer, reached through an address-space-3 pointer.
    Smem = 1,
}

impl MemorySpace {
    /// The keyword inside the fragment type (`local` or `smem`).
    pub const fn name(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Smem => "smem",
        }
    }

    /// The memory space a bytecode enum value decodes to.
    pub const fn from_u64(value: u64) -> Option<Self> {
        match value {
            0 => Some(Self::Local),
            1 => Some(Self::Smem),
            _ => None,
        }
    }
}

/// Widest cooperative synchronization domain a SIMT implementation uses.
/// It constrains the fragmap participant count: `warp` needs a positive
/// multiple of 32, `warpgroup` of 128, and `cta` the calling kernel's full
/// thread count (checked by the consumer, which knows the kernel).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum Synchronization {
    /// No cooperative synchronization. An implementation without a
    /// `synchronization` clause means the same thing; the clause carrying
    /// this value is written and printed only when it is present, see
    /// [`ExternImpl::synchronization`](crate::tilelib::ExternImpl::synchronization).
    #[default]
    None = 0,
    /// May synchronize within a warp.
    Warp = 1,
    /// May synchronize within a warpgroup of four warps.
    Warpgroup = 2,
    /// May synchronize across the whole CTA.
    Cta = 3,
}

impl Synchronization {
    /// The keyword inside `#cuda_tile.preview$synchronization<...>`.
    pub const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Warp => "warp",
            Self::Warpgroup => "warpgroup",
            Self::Cta => "cta",
        }
    }

    /// The synchronization domain a bytecode enum value decodes to.
    pub const fn from_u64(value: u64) -> Option<Self> {
        match value {
            0 => Some(Self::None),
            1 => Some(Self::Warp),
            2 => Some(Self::Warpgroup),
            3 => Some(Self::Cta),
            _ => None,
        }
    }
}

/// A symbolic integer, type or string expression (`SymbolicExprAttr`).
///
/// Layouts, scratch requests, constant arguments and fragment alignments are
/// built from these. v1 libraries use concrete integers almost everywhere;
/// the other variants exist so every wire form can be read back and so a
/// parametric library can be described later without a model change.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SymExpr {
    /// `#cuda_tile.preview$expr_int<v>`
    Int(i64),
    /// A constant parameter in scope, `#cuda_tile.preview$expr_var<"M">`.
    Var(String),
    /// An execution-context value such as `context.num_threads`, resolved
    /// by the consumer; its type is always `index`. Boxed because a type can
    /// in turn hold expressions (a fragment's alignment).
    Builtin {
        name: String,
        ty: Box<Type>,
    },
    /// `#cuda_tile.preview$expr_const_str<"...">`, a string constant argument.
    ConstStr(String),
    /// A concrete type used as a datatype argument (`f32`).
    ConcreteType(Box<Type>),
    /// A (possibly nested) CuTe mode tuple.
    Tuple(Vec<SymExpr>),
    Mul(Box<SymExpr>, Box<SymExpr>),
    /// Toward-zero division; rejected in integer contexts by the upstream
    /// verifier, kept so the wire form round-trips.
    Div(Box<SymExpr>, Box<SymExpr>),
    Add(Box<SymExpr>, Box<SymExpr>),
    Sub(Box<SymExpr>, Box<SymExpr>),
    /// `lhs / rhs` in the compact syntax.
    FloorDiv(Box<SymExpr>, Box<SymExpr>),
    /// `lhs /+ rhs` in the compact syntax.
    CeilDiv(Box<SymExpr>, Box<SymExpr>),
    /// `-operand`.
    Neg(Box<SymExpr>),
}

impl SymExpr {
    /// A reference to a declared constant parameter.
    pub fn var(name: impl Into<String>) -> Self {
        Self::Var(name.into())
    }

    /// A nested mode `(a, b, ...)`, as used inside layouts.
    pub fn tuple(values: impl IntoIterator<Item = SymExpr>) -> Self {
        Self::Tuple(values.into_iter().collect())
    }

    /// `context.<name>`, always `index`-typed.
    pub fn builtin(name: impl Into<String>) -> Self {
        Self::Builtin {
            name: name.into(),
            ty: Box::new(Type::Index),
        }
    }

    /// A concrete element type given as a constant argument.
    pub fn concrete_type(ty: Type) -> Self {
        Self::ConcreteType(Box::new(ty))
    }

    /// A string constant argument.
    pub fn const_str(value: impl Into<String>) -> Self {
        Self::ConstStr(value.into())
    }

    /// `lhs * rhs`.
    pub fn times(lhs: SymExpr, rhs: SymExpr) -> Self {
        Self::Mul(Box::new(lhs), Box::new(rhs))
    }

    /// `lhs + rhs`.
    pub fn plus(lhs: SymExpr, rhs: SymExpr) -> Self {
        Self::Add(Box::new(lhs), Box::new(rhs))
    }

    /// `lhs - rhs`.
    pub fn minus(lhs: SymExpr, rhs: SymExpr) -> Self {
        Self::Sub(Box::new(lhs), Box::new(rhs))
    }

    /// `lhs / rhs`, rounding toward negative infinity.
    pub fn floor_div(lhs: SymExpr, rhs: SymExpr) -> Self {
        Self::FloorDiv(Box::new(lhs), Box::new(rhs))
    }

    /// `lhs /+ rhs`, rounding toward positive infinity.
    pub fn ceil_div(lhs: SymExpr, rhs: SymExpr) -> Self {
        Self::CeilDiv(Box::new(lhs), Box::new(rhs))
    }

    /// `-operand`.
    pub fn negate(operand: SymExpr) -> Self {
        Self::Neg(Box::new(operand))
    }

    /// `true` when no variable or execution-context builtin is referenced,
    /// so every integer leaf folds to a value.
    pub fn is_concrete(&self) -> bool {
        match self {
            Self::Int(_) | Self::ConstStr(_) | Self::ConcreteType(_) => true,
            Self::Var(_) | Self::Builtin { .. } => false,
            Self::Tuple(values) => values.iter().all(Self::is_concrete),
            Self::Mul(l, r)
            | Self::Div(l, r)
            | Self::Add(l, r)
            | Self::Sub(l, r)
            | Self::FloorDiv(l, r)
            | Self::CeilDiv(l, r) => l.is_concrete() && r.is_concrete(),
            Self::Neg(x) => x.is_concrete(),
        }
    }

    /// Fold a concrete integer expression. `None` for symbolic leaves,
    /// tuples, non-integer leaves, division by zero and overflow, which is
    /// also what the upstream `simplify` refuses to fold.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            Self::Var(_)
            | Self::Builtin { .. }
            | Self::ConstStr(_)
            | Self::ConcreteType(_)
            | Self::Tuple(_) => None,
            Self::Mul(l, r) => l.as_int()?.checked_mul(r.as_int()?),
            Self::Add(l, r) => l.as_int()?.checked_add(r.as_int()?),
            Self::Sub(l, r) => l.as_int()?.checked_sub(r.as_int()?),
            Self::Div(l, r) => {
                let (l, r) = (l.as_int()?, r.as_int()?);
                if r == 0 {
                    None
                } else {
                    l.checked_div(r)
                }
            }
            Self::FloorDiv(l, r) => {
                let (l, r) = (l.as_int()?, r.as_int()?);
                if r == 0 {
                    None
                } else {
                    l.checked_div_euclid(r).map(|q| {
                        // Euclidean and floor division differ for a negative
                        // divisor with a nonzero remainder.
                        if r < 0 && l.rem_euclid(r) != 0 {
                            q - 1
                        } else {
                            q
                        }
                    })
                }
            }
            Self::CeilDiv(l, r) => {
                let (l, r) = (l.as_int()?, r.as_int()?);
                if r == 0 {
                    return None;
                }
                let q = l.checked_div(r)?;
                let rem = l.checked_rem(r)?;
                if rem != 0 && ((rem > 0) == (r > 0)) {
                    q.checked_add(1)
                } else {
                    Some(q)
                }
            }
            Self::Neg(x) => x.as_int()?.checked_neg(),
        }
    }

    /// Every type mentioned by a leaf, for type-table registration.
    pub fn types(&self) -> Vec<&Type> {
        let mut out = Vec::new();
        self.collect_types(&mut out);
        out
    }

    fn collect_types<'a>(&'a self, out: &mut Vec<&'a Type>) {
        match self {
            Self::Builtin { ty, .. } | Self::ConcreteType(ty) => out.push(ty),
            Self::Tuple(values) => values.iter().for_each(|v| v.collect_types(out)),
            Self::Mul(l, r)
            | Self::Div(l, r)
            | Self::Add(l, r)
            | Self::Sub(l, r)
            | Self::FloorDiv(l, r)
            | Self::CeilDiv(l, r) => {
                l.collect_types(out);
                r.collect_types(out);
            }
            Self::Neg(x) => x.collect_types(out),
            Self::Int(_) | Self::Var(_) | Self::ConstStr(_) => {}
        }
    }

    /// Every string a leaf interns (variable, builtin and constant names).
    pub fn strings(&self) -> Vec<&str> {
        let mut out = Vec::new();
        self.collect_strings(&mut out);
        out
    }

    fn collect_strings<'a>(&'a self, out: &mut Vec<&'a str>) {
        match self {
            Self::Var(name) | Self::Builtin { name, .. } | Self::ConstStr(name) => {
                out.push(name.as_str())
            }
            Self::Tuple(values) => values.iter().for_each(|v| v.collect_strings(out)),
            Self::Mul(l, r)
            | Self::Div(l, r)
            | Self::Add(l, r)
            | Self::Sub(l, r)
            | Self::FloorDiv(l, r)
            | Self::CeilDiv(l, r) => {
                l.collect_strings(out);
                r.collect_strings(out);
            }
            Self::Neg(x) => x.collect_strings(out),
            Self::Int(_) | Self::ConcreteType(_) => {}
        }
    }
}

/// A CuTe layout `(shapes):(strides)`; each mode may itself be a tuple
/// (`SymExpr::Tuple`), and shapes and strides nest identically.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Layout {
    /// Extent of every mode, outermost first.
    pub shapes: Vec<SymExpr>,
    /// Stride of every mode, nested the same way as `shapes`.
    pub strides: Vec<SymExpr>,
}

impl Layout {
    /// A flat layout from integer shapes and strides.
    pub fn flat(shapes: &[i64], strides: &[i64]) -> Self {
        Self {
            shapes: shapes.iter().copied().map(SymExpr::Int).collect(),
            strides: strides.iter().copied().map(SymExpr::Int).collect(),
        }
    }

    /// Number of top-level modes.
    pub fn rank(&self) -> usize {
        self.shapes.len()
    }

    /// `true` when every shape and stride folds to an integer.
    pub fn is_concrete(&self) -> bool {
        self.shapes
            .iter()
            .chain(&self.strides)
            .all(SymExpr::is_concrete)
    }

    // The fragment layouts a tile library needs: the `mem_layout` of a
    // buffer (`scalar_mem`, `contiguous_mem`) and the fragmaps that place
    // invocations and participants on a tile.

    /// `():()`: the rank-0 buffer of a by-value scalar. This crate narrows
    /// its use to `in` arguments in local memory, the one case the SIMT ABI
    /// promotes to a by-value scalar; upstream accepts a rank-0 layout on
    /// any fragment and passes an `out` or `inout` one as a pointer.
    pub fn scalar_mem() -> Self {
        Self::flat(&[], &[])
    }

    /// `(F):(1)`: a contiguous footprint of `footprint` elements.
    pub fn contiguous_mem(footprint: i64) -> Self {
        Self::flat(&[footprint], &[1])
    }

    /// `(N, 1, 1):(1, 0, 0)`: one invocation per tile element, no
    /// cooperating participants.
    pub fn elementwise(n: i64) -> Self {
        Self::flat(&[n, 1, 1], &[1, 0, 0])
    }

    /// `(N/32, 32, 1):(32, 1, 0)`: one element per lane, a whole warp per
    /// invocation.
    ///
    /// # Panics
    ///
    /// Panics when `n` is not a positive multiple of 32: the map would
    /// silently cover fewer elements than the tile holds.
    pub fn warp_lanes(n: i64) -> Self {
        assert!(
            n > 0 && n % 32 == 0,
            "Layout::warp_lanes: the tile size ({n}) must be a positive multiple of 32 lanes"
        );
        Self::flat(&[n / 32, 32, 1], &[32, 1, 0])
    }

    /// `(N/F, 1, F):(F, 0, 1)`: contiguous footprints of `f` elements per
    /// invocation.
    ///
    /// # Panics
    ///
    /// Panics when `f` is not positive or does not divide `n`: the
    /// footprints would not tile the operand.
    pub fn footprint_blocks(n: i64, f: i64) -> Self {
        assert!(
            f > 0,
            "Layout::footprint_blocks: the footprint ({f}) must be positive"
        );
        assert!(
            n % f == 0,
            "Layout::footprint_blocks: the footprint ({f}) must divide the tile size ({n})"
        );
        Self::flat(&[n / f, 1, f], &[f, 0, 1])
    }

    /// `(1, T, N):(0, 0, 1)`: the whole tile in one shared buffer seen by
    /// `threads` participants.
    pub fn shared_whole(threads: i64, n: i64) -> Self {
        Self::flat(&[1, threads, n], &[0, 0, 1])
    }
}

/// Implementation-local shared-memory scratch, in bytes
/// (`scratch { smem.size = ..., smem.alignment = ... }`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ScratchRequest {
    /// Bytes of shared memory the implementation needs per call.
    pub size: SymExpr,
    /// Alignment of that buffer in bytes, a power of two.
    pub alignment: SymExpr,
}

impl ScratchRequest {
    /// Upstream's default when `smem.alignment` is omitted.
    pub const DEFAULT_ALIGNMENT: i64 = 128;

    /// A request with an explicit alignment.
    pub fn new(size: SymExpr, alignment: SymExpr) -> Self {
        Self { size, alignment }
    }

    /// A fixed request with the default 128-byte alignment.
    pub fn bytes(size: i64) -> Self {
        Self::new(SymExpr::Int(size), SymExpr::Int(Self::DEFAULT_ALIGNMENT))
    }
}

/// A compile-time constant parameter declaration, `M: index`. The wildcard
/// `?` (name `"?"`, type `none`) marks an implementation slot that is not
/// specialized.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConstParam {
    /// The parameter name referenced by `SymExpr::Var`.
    pub name: String,
    /// `index`, `none` for a wildcard, or a preview parameter type.
    pub ty: Type,
}

impl ConstParam {
    /// The name of a wildcard slot.
    pub const WILDCARD_NAME: &'static str = "?";

    /// A parameter with an explicit type.
    pub fn new(name: impl Into<String>, ty: Type) -> Self {
        Self {
            name: name.into(),
            ty,
        }
    }

    /// An `index`-typed parameter, the common case.
    pub fn index(name: impl Into<String>) -> Self {
        Self::new(name, Type::Index)
    }

    /// An unspecialized implementation slot, printed as `?`.
    pub fn wildcard() -> Self {
        Self::new(Self::WILDCARD_NAME, Type::NoneType)
    }

    /// `true` for the `?` slot.
    pub fn is_wildcard(&self) -> bool {
        self.name == Self::WILDCARD_NAME && self.ty == Type::NoneType
    }
}

// ---------------------------------------------------------------------------
// Unified attribute enum
// ---------------------------------------------------------------------------

/// An IR attribute value.
///
/// `Integer` and `Float` carry their scalar type, matching MLIR's
/// `IntegerAttr` (value + IntegerType) and `FloatAttr` (value + FloatType).
#[derive(Debug, Clone, PartialEq)]
pub enum Attribute {
    Integer(i64, Type),
    Float(f64, Type),
    Bool(bool),
    Type(Type),
    String(String),
    Array(Vec<Attribute>),
    DenseElements(DenseElements),
    /// Dense array of i32 values, matching MLIR's `DenseI32ArrayAttr`; the
    /// encoding of op attributes declared with that type (`permutation`).
    ///
    /// Not for `operandSegmentSizes`: the bytecode writer reads the operand
    /// group sizes only from an [`Attribute::Array`] of [`Attribute::Integer`]s
    /// (see `operand_segment_sizes` in `bytecode/op_writer.rs`), and the
    /// attribute itself is never serialized — it only drives how the
    /// operands are grouped. A `DenseI32Array` there is silently treated as
    /// "one operand per group".
    DenseI32Array(Vec<i32>),
    DivBy(DivBy),
    SameElements(SameElements),
    Dictionary(Vec<(String, Attribute)>),
    OptimizationHints(OptimizationHints),
    Bounded(Bounded),
    // Public-preview SIMT interop attributes. A module carrying any of them
    // is written at `BytecodeVersion::PREVIEW_CONTENT` with the preview tag.
    Access(Access),
    MemorySpace(MemorySpace),
    Synchronization(Synchronization),
    Layout(Layout),
    /// Any symbolic expression, a bare mode tuple included
    /// (`SymExpr::Tuple` carries the `SymbolicTuple` tag on the wire).
    SymExpr(SymExpr),
    ScratchRequest(ScratchRequest),
    ConstParam(ConstParam),
}

impl Attribute {
    /// A public-preview attribute, or a container holding one (the access
    /// dictionaries of an `arg_attrs` array, for example).
    pub fn contains_preview(&self) -> bool {
        match self {
            Self::Access(_)
            | Self::MemorySpace(_)
            | Self::Synchronization(_)
            | Self::Layout(_)
            | Self::SymExpr(_)
            | Self::ScratchRequest(_)
            | Self::ConstParam(_) => true,
            Self::Type(ty) => ty.is_preview(),
            Self::Array(items) => items.iter().any(Self::contains_preview),
            Self::Dictionary(entries) => entries.iter().any(|(_, v)| v.contains_preview()),
            Self::OptimizationHints(hints) => hints
                .entries
                .iter()
                .any(|(_, values)| values.iter().any(|(_, v)| v.contains_preview())),
            Self::Integer(_, ty) | Self::Float(_, ty) => ty.is_preview(),
            Self::DenseElements(d) => d.element_type.is_preview(),
            Self::Bool(_)
            | Self::String(_)
            | Self::DenseI32Array(_)
            | Self::DivBy(_)
            | Self::SameElements(_)
            | Self::Bounded(_) => false,
        }
    }

    /// The per-argument dictionary `{cuda_tile.preview$access = ...}`.
    pub fn access_dict(access: Access) -> Self {
        Self::Dictionary(vec![(ACCESS_ATTR.to_string(), Self::Access(access))])
    }
    /// Dense boolean op attribute, e.g. the per-index `inbounds` vector.
    pub fn dense_bool_array(values: impl IntoIterator<Item = bool>) -> Self {
        Self::Array(values.into_iter().map(Self::Bool).collect())
    }

    /// Create a typed integer attribute. Shorthand for `Attribute::Integer(v, Type::Scalar(ty))`.
    pub fn int(v: i64, ty: ScalarType) -> Self {
        Attribute::Integer(v, Type::Scalar(ty))
    }
    /// Create an i32 integer attribute (the most common case for enum-valued attrs).
    pub fn i32(v: i64) -> Self {
        Attribute::Integer(v, Type::Scalar(ScalarType::I32))
    }
    /// Create a typed float attribute. Shorthand for `Attribute::Float(v, Type::Scalar(ty))`.
    pub fn float(v: f64, ty: ScalarType) -> Self {
        Attribute::Float(v, Type::Scalar(ty))
    }
}

/// Dense element data for constant tensors.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseElements {
    pub element_type: Type,
    pub shape: Vec<i64>,
    pub data: Vec<u8>,
}
