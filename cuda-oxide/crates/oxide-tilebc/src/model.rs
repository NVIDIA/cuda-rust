/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The vocabulary of a tile library description: the tiles a declaration
//! takes, the fragments an implementation receives, and the clauses that
//! tell the Tile compiler how to call the native function.
//!
//! Extents and layout modes are [`Expr`]s. A static library names concrete
//! extents; a parametric declaration names a constant parameter such as `N`
//! and its implementation's fragmaps are expressions over it, which the Tile
//! compiler instantiates for every call.

use std::fmt;

/// A Tile IR bytecode version (`major.minor`). The header tag bits are not
/// part of a version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BytecodeVersion {
    /// Major version, 13 for every Tile IR release so far.
    pub major: u8,
    /// Minor version.
    pub minor: u8,
}

impl BytecodeVersion {
    /// The first version with public-preview content, and the version the
    /// descriptions are written at when no assembler can be asked.
    pub const PREVIEW_FIRST: Self = Self {
        major: 13,
        minor: 5,
    };

    /// A version from its two numbers.
    pub const fn new(major: u8, minor: u8) -> Self {
        Self { major, minor }
    }

    /// Parse one line of `tileiras --list-versions`: `13.5` or `13.5.1`
    /// (the header tag is ignored). Lines that are not versions, such as a
    /// named build an assembler may list, give `None`.
    pub fn parse(text: &str) -> Option<Self> {
        let mut parts = text.trim().split('.');
        let major = parts.next()?.parse().ok()?;
        let minor = parts.next()?.parse().ok()?;
        if let Some(tag) = parts.next() {
            tag.parse::<u16>().ok()?;
        }
        if parts.next().is_some() {
            return None;
        }
        Some(Self { major, minor })
    }
}

impl fmt::Display for BytecodeVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// Element types a description can name. Tile IR has no unsigned integers:
/// the unsigned Rust types are the signless `iN` of the same width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    /// `f16`
    F16,
    /// `bf16`
    BF16,
    /// `f32`
    F32,
    /// `f64`
    F64,
    /// `i8`
    I8,
    /// `i16`
    I16,
    /// `i32`
    I32,
    /// `i64`
    I64,
}

impl ScalarType {
    /// The type-table tag.
    pub(crate) fn tag(self) -> u8 {
        match self {
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 3,
            Self::I64 => 4,
            Self::F16 => 5,
            Self::BF16 => 6,
            Self::F32 => 7,
            Self::F64 => 9,
        }
    }

    /// The textual name.
    pub fn name(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::I8 => "i8",
            Self::I16 => "i16",
            Self::I32 => "i32",
            Self::I64 => "i64",
        }
    }
}

/// A compile-time integer expression: a literal, a declaration's constant
/// parameter, a compiler builtin such as `context.num_threads`, or
/// arithmetic over those.
///
/// Division is the flooring kind only; Tile IR's truncating division is not
/// accepted in the integer positions a library uses.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expr {
    /// A literal.
    Int(i64),
    /// A constant parameter of the enclosing declaration or implementation.
    Var(String),
    /// A compiler builtin in the `context.` namespace, of type `index`.
    Builtin(String),
    /// `lhs * rhs`
    Mul(Box<Expr>, Box<Expr>),
    /// `lhs + rhs`
    Add(Box<Expr>, Box<Expr>),
    /// `lhs - rhs`
    Sub(Box<Expr>, Box<Expr>),
    /// `-x`
    Neg(Box<Expr>),
    /// `lhs / rhs`, rounding toward negative infinity.
    FloorDiv(Box<Expr>, Box<Expr>),
    /// `lhs /+ rhs`, rounding toward positive infinity.
    CeilDiv(Box<Expr>, Box<Expr>),
}

impl From<i64> for Expr {
    fn from(value: i64) -> Self {
        Expr::Int(value)
    }
}

impl From<&str> for Expr {
    fn from(name: &str) -> Self {
        Expr::Var(name.to_owned())
    }
}

impl Expr {
    /// A constant parameter by name.
    pub fn var(name: impl Into<String>) -> Self {
        Expr::Var(name.into())
    }

    /// The literal value, when the expression is one.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Expr::Int(value) => Some(*value),
            _ => None,
        }
    }

    /// `self * rhs`, folded when both sides are literals.
    pub fn times(self, rhs: impl Into<Expr>) -> Self {
        let rhs = rhs.into();
        match (&self, &rhs) {
            (Expr::Int(a), Expr::Int(b)) => Expr::Int(a * b),
            _ => Expr::Mul(Box::new(self), Box::new(rhs)),
        }
    }

    /// `self / rhs` rounding down, folded when both sides are literals.
    pub fn floor_div(self, rhs: impl Into<Expr>) -> Self {
        let rhs = rhs.into();
        match (&self, &rhs) {
            (Expr::Int(a), Expr::Int(b)) if *b != 0 => Expr::Int(a.div_euclid(*b)),
            _ => Expr::FloorDiv(Box::new(self), Box::new(rhs)),
        }
    }

    /// Every constant parameter the expression names, in first-use order.
    pub fn vars(&self, out: &mut Vec<String>) {
        match self {
            Expr::Int(_) | Expr::Builtin(_) => {}
            Expr::Var(name) => {
                if !out.iter().any(|seen| seen == name) {
                    out.push(name.clone());
                }
            }
            Expr::Neg(x) => x.vars(out),
            Expr::Mul(l, r)
            | Expr::Add(l, r)
            | Expr::Sub(l, r)
            | Expr::FloorDiv(l, r)
            | Expr::CeilDiv(l, r) => {
                l.vars(out);
                r.vars(out);
            }
        }
    }

    /// Whether the expression may stand in a tile dimension, which admits
    /// no division.
    pub fn is_dimension(&self) -> bool {
        match self {
            Expr::Int(_) | Expr::Var(_) | Expr::Builtin(_) => true,
            Expr::Neg(x) => x.is_dimension(),
            Expr::Mul(l, r) | Expr::Add(l, r) | Expr::Sub(l, r) => {
                l.is_dimension() && r.is_dimension()
            }
            Expr::FloorDiv(..) | Expr::CeilDiv(..) => false,
        }
    }

    /// Evaluate with `bind` supplying every parameter's value. Builtins have
    /// no value here.
    pub fn eval(&self, bind: &dyn Fn(&str) -> Option<i64>) -> Result<i64, String> {
        Ok(match self {
            Expr::Int(value) => *value,
            Expr::Var(name) => bind(name).ok_or_else(|| format!("`{name}` has no value"))?,
            Expr::Builtin(name) => {
                return Err(format!("`{name}` has no value before the kernel compiles"));
            }
            Expr::Neg(x) => -x.eval(bind)?,
            Expr::Mul(l, r) => l.eval(bind)? * r.eval(bind)?,
            Expr::Add(l, r) => l.eval(bind)? + r.eval(bind)?,
            Expr::Sub(l, r) => l.eval(bind)? - r.eval(bind)?,
            Expr::FloorDiv(l, r) | Expr::CeilDiv(l, r) => {
                let divisor = r.eval(bind)?;
                if divisor == 0 {
                    return Err(format!("`{self}` divides by zero"));
                }
                let dividend = l.eval(bind)?;
                let floor = dividend.div_euclid(divisor);
                if matches!(self, Expr::FloorDiv(..)) || dividend.rem_euclid(divisor) == 0 {
                    floor
                } else {
                    floor + 1
                }
            }
        })
    }

    fn precedence(&self) -> u8 {
        match self {
            Expr::Int(_) | Expr::Var(_) | Expr::Builtin(_) | Expr::Neg(_) => 3,
            Expr::Mul(..) | Expr::FloorDiv(..) | Expr::CeilDiv(..) => 2,
            Expr::Add(..) | Expr::Sub(..) => 1,
        }
    }

    fn fmt_operand(&self, operand: &Expr, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if operand.precedence() < self.precedence() {
            write!(f, "({operand})")
        } else {
            write!(f, "{operand}")
        }
    }
}

/// The compact Tile IR spelling: `N`, `N * 4`, `N / 32`, `N /+ 32`, `-N`.
impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let binary = |f: &mut fmt::Formatter<'_>, l: &Expr, op: &str, r: &Expr| {
            self.fmt_operand(l, f)?;
            write!(f, " {op} ")?;
            self.fmt_operand(r, f)
        };
        match self {
            Expr::Int(value) => write!(f, "{value}"),
            Expr::Var(name) | Expr::Builtin(name) => write!(f, "{name}"),
            Expr::Neg(x) => {
                write!(f, "-")?;
                self.fmt_operand(x, f)
            }
            Expr::Mul(l, r) => binary(f, l, "*", r),
            Expr::Add(l, r) => binary(f, l, "+", r),
            Expr::Sub(l, r) => binary(f, l, "-", r),
            Expr::FloorDiv(l, r) => binary(f, l, "/", r),
            Expr::CeilDiv(l, r) => binary(f, l, "/+", r),
        }
    }
}

/// A compile-time constant parameter of a declaration or implementation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConstParam {
    /// A named integer parameter, `N: index`.
    Index(String),
    /// `?`: an implementation slot that binds whatever the call supplies.
    /// Declarations cannot have one.
    Wildcard,
}

impl ConstParam {
    /// An integer parameter by name.
    pub fn index(name: impl Into<String>) -> Self {
        ConstParam::Index(name.into())
    }

    /// The name, `?` for the wildcard.
    pub fn name(&self) -> &str {
        match self {
            ConstParam::Index(name) => name,
            ConstParam::Wildcard => "?",
        }
    }

    /// Whether this is the wildcard slot.
    pub fn is_wildcard(&self) -> bool {
        matches!(self, ConstParam::Wildcard)
    }
}

/// A tile type, the value a declaration argument carries. Its extents are
/// literals in a static declaration and expressions over the declaration's
/// constant parameters in a parametric one.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TileType {
    /// Extents, one per dimension.
    pub shape: Vec<Expr>,
    /// Element type.
    pub element: ScalarType,
}

impl TileType {
    /// A tile of concrete `shape` by `element`.
    pub fn new(shape: Vec<i64>, element: ScalarType) -> Self {
        Self {
            shape: shape.into_iter().map(Expr::Int).collect(),
            element,
        }
    }

    /// A tile whose extents are expressions.
    pub fn symbolic(shape: Vec<Expr>, element: ScalarType) -> Self {
        Self { shape, element }
    }

    /// Whether every extent is a literal.
    pub fn is_concrete(&self) -> bool {
        self.shape.iter().all(|dim| dim.as_int().is_some())
    }

    /// Number of elements, when every extent is a literal.
    pub fn concrete_size(&self) -> Option<i64> {
        self.shape.iter().map(Expr::as_int).product()
    }
}

/// How a callee uses an argument. `out` and `inout` arguments are also the
/// declaration's results, in argument order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Access {
    /// Read only.
    In,
    /// Written, not read.
    Out,
    /// Read and written.
    InOut,
}

impl Access {
    /// The bytecode value.
    pub(crate) fn value(self) -> u64 {
        match self {
            Self::In => 0,
            Self::Out => 1,
            Self::InOut => 2,
        }
    }

    /// The textual name.
    pub fn name(self) -> &'static str {
        match self {
            Self::In => "in",
            Self::Out => "out",
            Self::InOut => "inout",
        }
    }

    /// Whether the callee writes the argument.
    pub fn is_written(self) -> bool {
        !matches!(self, Self::In)
    }
}

/// Where a fragment's buffer lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemorySpace {
    /// Per-thread storage.
    Local,
    /// CTA shared memory.
    Smem,
}

impl MemorySpace {
    pub(crate) fn value(self) -> u64 {
        match self {
            Self::Local => 0,
            Self::Smem => 1,
        }
    }

    /// The textual name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::Smem => "smem",
        }
    }
}

/// The threads that call the native function together.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Synchronization {
    /// One thread per invocation.
    None,
    /// The 32 lanes of a warp.
    Warp,
    /// Four warps.
    Warpgroup,
    /// The whole CTA.
    Cta,
}

impl Synchronization {
    pub(crate) fn value(self) -> u64 {
        match self {
            Self::None => 0,
            Self::Warp => 1,
            Self::Warpgroup => 2,
            Self::Cta => 3,
        }
    }

    /// The textual name.
    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Warp => "warp",
            Self::Warpgroup => "warpgroup",
            Self::Cta => "cta",
        }
    }

    /// How many threads one invocation spans.
    pub fn participants(self) -> i64 {
        match self {
            Self::None => 1,
            Self::Warp => 32,
            Self::Warpgroup => 128,
            Self::Cta => 0,
        }
    }
}

/// A layout `(shapes):(strides)`; every mode is an [`Expr`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Layout {
    /// Mode extents.
    pub shapes: Vec<Expr>,
    /// Mode strides.
    pub strides: Vec<Expr>,
}

impl Layout {
    /// A concrete layout from its shapes and strides.
    pub fn flat(shapes: &[i64], strides: &[i64]) -> Self {
        Self {
            shapes: shapes.iter().copied().map(Expr::Int).collect(),
            strides: strides.iter().copied().map(Expr::Int).collect(),
        }
    }

    /// A layout whose modes are expressions.
    pub fn symbolic(shapes: Vec<Expr>, strides: Vec<Expr>) -> Self {
        Self { shapes, strides }
    }

    /// The rank-0 buffer layout `():()` of a scalar passed by value.
    pub fn scalar_mem() -> Self {
        Self::flat(&[], &[])
    }

    /// The buffer layout `(F):(1)` of `footprint` contiguous elements.
    pub fn contiguous_mem(footprint: i64) -> Self {
        Self::flat(&[footprint], &[1])
    }

    /// One invocation per element, no participants: `(N, 1, 1):(1, 0, 0)`.
    pub fn elementwise(tile_size: impl Into<Expr>) -> Self {
        Self::symbolic(
            vec![tile_size.into(), Expr::Int(1), Expr::Int(1)],
            vec![Expr::Int(1), Expr::Int(0), Expr::Int(0)],
        )
    }

    /// Warp-synchronous, one element per lane: `(N / 32, 32, 1):(32, 1, 0)`.
    pub fn warp_lanes(tile_size: impl Into<Expr>) -> Self {
        Self::symbolic(
            vec![tile_size.into().floor_div(32), Expr::Int(32), Expr::Int(1)],
            vec![Expr::Int(32), Expr::Int(1), Expr::Int(0)],
        )
    }

    /// One invocation per block of `footprint` contiguous elements:
    /// `(N / F, 1, F):(F, 0, 1)`.
    pub fn footprint_blocks(tile_size: impl Into<Expr>, footprint: i64) -> Self {
        Self::symbolic(
            vec![
                tile_size.into().floor_div(footprint),
                Expr::Int(1),
                Expr::Int(footprint),
            ],
            vec![Expr::Int(footprint), Expr::Int(0), Expr::Int(1)],
        )
    }

    /// Warp-synchronous with a footprint of `F` per lane:
    /// `(N / (32 F), 32, F):(32 F, F, 1)`.
    pub fn warp_footprints(tile_size: impl Into<Expr>, footprint: i64) -> Self {
        let per_warp = 32 * footprint;
        Self::symbolic(
            vec![
                tile_size.into().floor_div(per_warp),
                Expr::Int(32),
                Expr::Int(footprint),
            ],
            vec![Expr::Int(per_warp), Expr::Int(footprint), Expr::Int(1)],
        )
    }

    /// Number of modes.
    pub fn rank(&self) -> usize {
        self.shapes.len()
    }

    /// Whether every mode is a literal.
    pub fn is_concrete(&self) -> bool {
        self.shapes
            .iter()
            .chain(&self.strides)
            .all(|mode| mode.as_int().is_some())
    }

    /// Product of the shapes, when every shape is a literal.
    pub fn concrete_size(&self) -> Option<i64> {
        self.shapes.iter().map(Expr::as_int).product()
    }

    /// Every constant parameter the layout names.
    pub fn vars(&self, out: &mut Vec<String>) {
        for mode in self.shapes.iter().chain(&self.strides) {
            mode.vars(out);
        }
    }
}

/// The fragment the native function receives for one declaration argument.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FragmentType {
    /// Elements one invocation owns (its footprint), one extent per dimension.
    pub shape: Vec<i64>,
    /// Element type.
    pub element: ScalarType,
    /// Where the buffer lives.
    pub memory_space: MemorySpace,
    /// How the footprint sits in the buffer; rank 0 means passed by value.
    pub mem_layout: Layout,
    /// `(invocations, participants, footprint):(strides)` onto the tile.
    pub fragmap: Layout,
}

impl FragmentType {
    /// A fragment from its parts. The alignment the bytecode carries is
    /// always the default of one.
    pub fn new(
        shape: Vec<i64>,
        element: ScalarType,
        memory_space: MemorySpace,
        mem_layout: Layout,
        fragmap: Layout,
    ) -> Self {
        Self {
            shape,
            element,
            memory_space,
            mem_layout,
            fragmap,
        }
    }

    /// Elements per invocation.
    pub fn footprint(&self) -> i64 {
        self.shape.iter().product()
    }

    /// Every constant parameter the fragment's layouts name.
    pub fn vars(&self, out: &mut Vec<String>) {
        self.mem_layout.vars(out);
        self.fragmap.vars(out);
    }
}

/// One argument of a declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileArg {
    /// Argument name, printed as `%name`.
    pub name: String,
    /// The tile at the call boundary.
    pub ty: TileType,
    /// How the callee uses it.
    pub access: Access,
}

impl TileArg {
    /// An argument from its parts.
    pub fn new(name: impl Into<String>, ty: TileType, access: Access) -> Self {
        Self {
            name: name.into(),
            ty,
            access,
        }
    }
}

/// One argument of an implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FragmentArg {
    /// Argument name, printed as `%name`.
    pub name: String,
    /// The fragment the native function receives.
    pub fragment: FragmentType,
    /// Must equal the access of the matching declaration argument.
    pub access: Access,
}

impl FragmentArg {
    /// An argument from its parts.
    pub fn new(name: impl Into<String>, fragment: FragmentType, access: Access) -> Self {
        Self {
            name: name.into(),
            fragment,
            access,
        }
    }
}

/// An extern declaration: what a Tile kernel can call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Declaration {
    /// The symbol a `preview$call` names.
    pub name: String,
    /// Constant parameters the call must supply, `<[N: index]>`; empty for
    /// a static declaration.
    pub const_params: Vec<ConstParam>,
    /// Arguments in order; at least one.
    pub args: Vec<TileArg>,
}

impl Declaration {
    /// A static declaration from its name and arguments.
    pub fn new(name: impl Into<String>, args: Vec<TileArg>) -> Self {
        Self {
            name: name.into(),
            const_params: Vec::new(),
            args,
        }
    }

    /// A parametric declaration: `@name<[params]>(args)`.
    pub fn parametric(
        name: impl Into<String>,
        const_params: Vec<ConstParam>,
        args: Vec<TileArg>,
    ) -> Self {
        Self {
            name: name.into(),
            const_params,
            args,
        }
    }

    /// The tiles the declaration returns: its written arguments, in order.
    pub fn results(&self) -> impl Iterator<Item = &TileType> {
        self.args
            .iter()
            .filter(|arg| arg.access.is_written())
            .map(|arg| &arg.ty)
    }
}

/// An implementation: the native symbol the code payload defines and how
/// its parameters map onto a declaration's tiles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Implementation {
    /// The linked symbol.
    pub symbol: String,
    /// The declaration this implements.
    pub implements: String,
    /// The implementation's own constant parameters, one per declaration
    /// parameter: the same name for a parameter the fragments use, `?` for
    /// a slot bound by the arguments below. Empty for a static one.
    pub const_params: Vec<ConstParam>,
    /// The arguments to the declaration's parameters, `implements
    /// @decl<[args]>`: the identity `N` for a named slot, any expression
    /// for a wildcard slot. Empty for a static implementation.
    pub const_args: Vec<Expr>,
    /// One fragment per declaration argument, in order.
    pub args: Vec<FragmentArg>,
    /// The `synchronization` clause; `None` leaves the clause out, which
    /// means one thread per invocation.
    pub synchronization: Option<Synchronization>,
    /// Target names such as `sm_120`; empty matches every target.
    pub arch: Vec<String>,
    /// `frontendABI` entries: printed type to native type.
    pub frontend_abi: Vec<(String, String)>,
}

impl Implementation {
    /// A static implementation with no optional clauses.
    pub fn new(
        symbol: impl Into<String>,
        implements: impl Into<String>,
        args: Vec<FragmentArg>,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            implements: implements.into(),
            const_params: Vec::new(),
            const_args: Vec::new(),
            args,
            synchronization: None,
            arch: Vec::new(),
            frontend_abi: Vec::new(),
        }
    }

    /// A parametric implementation that forwards every declaration
    /// parameter by name: `@symbol<[N: index]> implements @decl<[N]>`.
    pub fn parametric(
        symbol: impl Into<String>,
        implements: impl Into<String>,
        params: Vec<String>,
        args: Vec<FragmentArg>,
    ) -> Self {
        let mut imp = Self::new(symbol, implements, args);
        imp.const_args = params.iter().map(Expr::var).collect();
        imp.const_params = params.into_iter().map(ConstParam::Index).collect();
        imp
    }
}

/// A tile library's two halves, under one module name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Library {
    /// The `cuda_tile.module` name of both halves.
    pub name: String,
    /// The interface: what can be called.
    pub declarations: Vec<Declaration>,
    /// The catalog: how each declaration is implemented.
    pub implementations: Vec<Implementation>,
}

impl Library {
    /// A library from its parts.
    pub fn new(
        name: impl Into<String>,
        declarations: Vec<Declaration>,
        implementations: Vec<Implementation>,
    ) -> Self {
        Self {
            name: name.into(),
            declarations,
            implementations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expressions_fold_print_and_evaluate() {
        let n = Expr::var("N");
        assert_eq!(Expr::Int(128).floor_div(32), Expr::Int(4));
        assert_eq!(Expr::Int(3).times(4), Expr::Int(12));
        let lanes = n.clone().floor_div(32);
        assert_eq!(lanes.to_string(), "N / 32");
        assert_eq!(
            Expr::Mul(
                Box::new(Expr::Add(Box::new(n.clone()), Box::new(1.into()))),
                Box::new(2.into())
            )
            .to_string(),
            "(N + 1) * 2"
        );
        let bind = |name: &str| (name == "N").then_some(256);
        assert_eq!(lanes.eval(&bind).unwrap(), 8);
        assert_eq!(
            Expr::CeilDiv(Box::new(n.clone()), Box::new(100.into()))
                .eval(&bind)
                .unwrap(),
            3
        );
        assert!(Expr::var("M").eval(&bind).unwrap_err().contains("`M`"));
        assert!(n.is_dimension() && !lanes.is_dimension());
        let mut vars = Vec::new();
        Layout::warp_footprints(n, 4).vars(&mut vars);
        assert_eq!(vars, vec!["N".to_string()]);
    }

    #[test]
    fn concrete_layouts_are_unchanged() {
        assert_eq!(
            Layout::elementwise(128),
            Layout::flat(&[128, 1, 1], &[1, 0, 0])
        );
        assert_eq!(
            Layout::warp_lanes(128),
            Layout::flat(&[4, 32, 1], &[32, 1, 0])
        );
        assert_eq!(
            Layout::footprint_blocks(128, 4),
            Layout::flat(&[32, 1, 4], &[4, 0, 1])
        );
        assert_eq!(
            Layout::warp_footprints(256, 4),
            Layout::flat(&[2, 32, 4], &[128, 4, 1])
        );
        assert_eq!(Layout::warp_lanes(128).concrete_size(), Some(128));
        assert!(Layout::warp_lanes(Expr::var("N")).concrete_size().is_none());
    }
}
