/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The vocabulary of a tile library description: the tiles a declaration
//! takes, the fragments an implementation receives, and the clauses that
//! tell the Tile compiler how to call the native function.

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

/// A concrete tile type, the value a declaration argument carries.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TileType {
    /// Extents, one per dimension.
    pub shape: Vec<i64>,
    /// Element type.
    pub element: ScalarType,
}

impl TileType {
    /// A tile of `shape` by `element`.
    pub fn new(shape: Vec<i64>, element: ScalarType) -> Self {
        Self { shape, element }
    }

    /// Number of elements.
    pub fn size(&self) -> i64 {
        self.shape.iter().product()
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

/// A concrete layout: `(shapes):(strides)`, integers only.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Layout {
    /// Mode extents.
    pub shapes: Vec<i64>,
    /// Mode strides.
    pub strides: Vec<i64>,
}

impl Layout {
    /// A layout from its shapes and strides.
    pub fn flat(shapes: &[i64], strides: &[i64]) -> Self {
        Self {
            shapes: shapes.to_vec(),
            strides: strides.to_vec(),
        }
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
    pub fn elementwise(tile_size: i64) -> Self {
        Self::flat(&[tile_size, 1, 1], &[1, 0, 0])
    }

    /// Warp-synchronous, one element per lane: `(N/32, 32, 1):(32, 1, 0)`.
    pub fn warp_lanes(tile_size: i64) -> Self {
        Self::flat(&[tile_size / 32, 32, 1], &[32, 1, 0])
    }

    /// One invocation per block of `footprint` contiguous elements:
    /// `(N/F, 1, F):(F, 0, 1)`.
    pub fn footprint_blocks(tile_size: i64, footprint: i64) -> Self {
        Self::flat(&[tile_size / footprint, 1, footprint], &[footprint, 0, 1])
    }

    /// Number of dimensions.
    pub fn rank(&self) -> usize {
        self.shapes.len()
    }

    /// Product of the shapes.
    pub fn size(&self) -> i64 {
        self.shapes.iter().product()
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
    /// Arguments in order; at least one.
    pub args: Vec<TileArg>,
}

impl Declaration {
    /// A declaration from its name and arguments.
    pub fn new(name: impl Into<String>, args: Vec<TileArg>) -> Self {
        Self {
            name: name.into(),
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
    /// An implementation with no optional clauses.
    pub fn new(
        symbol: impl Into<String>,
        implements: impl Into<String>,
        args: Vec<FragmentArg>,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            implements: implements.into(),
            args,
            synchronization: None,
            arch: Vec::new(),
            frontend_abi: Vec::new(),
        }
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
