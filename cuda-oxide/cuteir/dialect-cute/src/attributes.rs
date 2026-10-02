/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed attributes for the cute dialect.
//!
//! Layouts are stored structurally (the `cute-layout` types), never as
//! strings; only the *textual syntax* is the CuTe notation in quotes, e.g.
//! `"(8,32):(32,1)"`, so module dumps read like CuTe IR.

use core::fmt;

use combine::Parser;
use cute_layout::{ComposedLayout, Layout, OffsetUnit, ParseLayoutError, Swizzle};
use pliron::common_traits::Verify;
use pliron::context::Context;
use pliron::impl_printable_for_display;
use pliron::irfmt::parsers::quoted_string_parser;
use pliron::parsable::{Parsable, ParseResult, StateStream};
use pliron::result::Error;
use pliron::verify_err_noloc;
use pliron_derive::pliron_attr;

/// A promise that an integer is divisible by this positive number.
///
/// This is deliberately a typed attribute instead of an unstructured string:
/// A later direct SSA consumer can ask the defining `cute.assume_div` op for
/// the exact divisor without rediscovering it from surrounding arithmetic.
#[pliron_attr(name = "cute.divisibility", format = "$0", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct CuteDivisibilityAttr(pub u64);

/// Width of one copy atom in bytes.
#[pliron_attr(name = "cute.copy_atom_bytes", format = "$0", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct CuteCopyAtomAttr(pub u32);

/// Which side of a staged load owns one ring position.
///
/// ```text
/// Producer  waits for an empty stage, then starts TMA
/// Consumer  waits for a full stage, then reads and releases it
/// ```
#[pliron_attr(name = "cute.pipeline_role", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CutePipelineRoleAttr {
    Producer,
    Consumer,
}

/// Compile-time facts attached to the ordinary `(slot, phase)` scalars.
///
/// The attribute says how to interpret those scalars. It is not a runtime
/// state object:
///
/// ```text
/// pipeline_state<Producer, 3>
///                    │
///                    └── slot: u32, phase: u32
/// ```
#[pliron_attr(name = "cute.pipeline_state", format = "`<` $role `,` $stages `>`")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct CutePipelineStateAttr {
    pub role: CutePipelineRoleAttr,
    pub stages: u64,
}

impl CutePipelineStateAttr {
    /// Describe producer or consumer state for one fixed-size stage ring.
    #[must_use]
    pub const fn new(role: CutePipelineRoleAttr, stages: u64) -> Self {
        Self { role, stages }
    }

    /// Describe producer state for one fixed-size stage ring.
    #[must_use]
    pub const fn producer(stages: u64) -> Self {
        Self::new(CutePipelineRoleAttr::Producer, stages)
    }

    /// Describe consumer state for one fixed-size stage ring.
    #[must_use]
    pub const fn consumer(stages: u64) -> Self {
        Self::new(CutePipelineRoleAttr::Consumer, stages)
    }
}

impl Verify for CutePipelineStateAttr {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if self.stages == 0 || self.stages > u64::from(u32::MAX) {
            return verify_err_noloc!(
                "cute.pipeline_state stages must be between 1 and {}, got {}",
                u32::MAX,
                self.stages
            );
        }
        Ok(())
    }
}

/// The logical output-tile grid owned by a static scheduler.
///
/// M changes first, then N, then batch (also called L):
///
/// ```text
/// grid<3, 2, 1>
///
/// linear:  0      1      2      3      4      5
/// tile:   0,0    1,0    2,0    0,1    1,1    2,1
/// ```
///
/// All three sizes are positive. Their product must fit in `u64`, which is
/// the scalar type carried through the scheduler loop.
#[pliron_attr(
    name = "cute.tile_grid",
    format = "`<` $m_tiles `,` $n_tiles `,` $batches `>`"
)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct CuteTileGridAttr {
    pub m_tiles: u64,
    pub n_tiles: u64,
    pub batches: u64,
}

impl CuteTileGridAttr {
    /// Describe a grid where M changes first, then N, then batch.
    #[must_use]
    pub const fn new(m_tiles: u64, n_tiles: u64, batches: u64) -> Self {
        Self {
            m_tiles,
            n_tiles,
            batches,
        }
    }

    /// Return the number of logical work tiles when multiplication is safe.
    #[must_use]
    pub const fn total_tiles(self) -> Option<u64> {
        let Some(mn) = self.m_tiles.checked_mul(self.n_tiles) else {
            return None;
        };
        mn.checked_mul(self.batches)
    }
}

impl Verify for CuteTileGridAttr {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if self.m_tiles == 0 || self.n_tiles == 0 || self.batches == 0 {
            return verify_err_noloc!("cute.tile_grid sizes must be greater than zero");
        }
        if self.total_tiles().is_none() {
            return verify_err_noloc!("cute.tile_grid tile count must fit in u64");
        }
        Ok(())
    }
}

/// Where a tensor view's storage lives.
///
/// The first tensor flow starts in global memory. TMA adds a shared-memory
/// destination without changing the meaning of the existing global variant:
///
/// ```text
/// host buffer ── kernel argument ──► global GPU memory
///                                      │ TMA
///                                      ▼
///                              CTA shared memory
/// ```
///
/// Keeping the choice typed avoids turning address-space numbers such as `1`
/// or `3` into a hidden contract.
#[pliron_attr(name = "cute.tensor_space", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CuteTensorAddressSpaceAttr {
    Gmem,
    /// CTA-local shared memory, visible to every thread in one block.
    Smem,
}

/// Whether a tensor view may only read or may also write.
///
/// ```text
/// ReadOnly   storage ──► registers
/// ReadWrite  storage ◄── registers
/// ```
#[pliron_attr(name = "cute.tensor_access", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CuteTensorAccessAttr {
    ReadOnly,
    ReadWrite,
}

/// How to read the bits stored by a tensor view.
///
/// `Plain` is the elementwise path: one `f16` or `f32` storage value already
/// is the logical value. The packed GEMV path keeps the other two meanings
/// visible:
///
/// ```text
/// E2M1   two four-bit values in global u8, or four in shared f16
/// UE8M0  one scale in global u8, or four packed scales in shared u32
/// ```
#[pliron_attr(name = "cute.tensor_format", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CuteTensorFormatAttr {
    Plain,
    E2M1,
    UE8M0,
}

/// Which logical row coordinate a tensor uses.
///
/// ```text
/// Mkl  rows follow M: A[m, k, l]
/// Nkl  rows follow N: B[n, k, l]
/// Generic             elementwise data
/// ```
///
/// `Nkl` is a single vector row when GEMV has `N = 1`, and a full B matrix
/// when GEMM has many N rows. Keeping the coordinate name avoids baking one
/// kernel profile into the tensor type.
///
/// The role is a compile-time fact. It does not change the stored bytes.
#[pliron_attr(name = "cute.tensor_role", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CuteTensorRoleAttr {
    Generic,
    Mkl,
    Nkl,
}

/// The one-dimensional layout carried by a v0 tensor view.
///
/// The tile width stays inside the layout instead of living in a nearby
/// untyped integer:
///
/// ```text
/// Contiguous1D       [0 1 2 3 4 5 6 7]
/// Zipped1D<4>        [0 1 2 3] [4 5 6 7]
/// Tile1D<4>           ^ one selected group ^
/// ```
#[pliron_attr(name = "cute.tensor_layout", format)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CuteTensorLayoutAttr {
    Contiguous1D,
    Zipped1D(u64),
    Tile1D(u64),
    /// Logical K values are adjacent. Packed formats may place more than one
    /// logical value in each storage element.
    KMajor,
    /// Blackwell's canonical scale order. The number says how many logical
    /// values share one scale. The fixed atom geometry is named by the
    /// `SM1XX_BLOCK_SCALE_*` constants in this module.
    BlockScaleKMajor(u64),
    /// A two-dimensional tile moved as raw carrier values by TMA.
    ///
    /// The exact row/column shape, stride, and swizzle live in the enclosing
    /// `cute.tma_view` type as a composed layout.
    Tma2D,
}

impl CuteTensorLayoutAttr {
    /// Return the tile width for a zipped or selected-tile layout.
    #[must_use]
    pub const fn tile_size(self) -> Option<u64> {
        match self {
            Self::Contiguous1D | Self::KMajor | Self::BlockScaleKMajor(_) | Self::Tma2D => None,
            Self::Zipped1D(size) | Self::Tile1D(size) => Some(size),
        }
    }

    /// Return how many logical K values share one scale, when this is the
    /// canonical block-scale layout.
    #[must_use]
    pub const fn values_per_scale(self) -> Option<u64> {
        match self {
            Self::BlockScaleKMajor(size) => Some(size),
            _ => None,
        }
    }
}

/// Logical rows inside one Blackwell canonical scale atom.
pub const SM1XX_BLOCK_SCALE_ROWS_PER_ATOM: u64 = 128;
/// Consecutive K-scale groups inside one Blackwell canonical scale atom.
pub const SM1XX_BLOCK_SCALE_GROUPS_PER_ATOM: u64 = 4;
/// Physical bytes reserved by one Blackwell canonical scale atom.
pub const SM1XX_BLOCK_SCALE_ATOM_BYTES: u64 = 512;

impl Verify for CuteTensorLayoutAttr {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if self.tile_size().is_some_and(|size| size == 0) {
            return verify_err_noloc!("cute.tensor_layout tile size must be greater than zero");
        }
        if matches!(self, Self::BlockScaleKMajor(0)) {
            return verify_err_noloc!(
                "cute.tensor_layout values per scale must be greater than zero"
            );
        }
        Ok(())
    }
}

/// Which part of a block-scaled tensor is currently selected.
///
/// ```text
/// Full ── select (batch, row) ──► Row ── select K tile ──► KTile<64>
/// ```
#[pliron_attr(name = "cute.scaled_layout", format)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CuteScaledLayoutAttr {
    Full,
    Row,
    KTile(u64),
}

impl Verify for CuteScaledLayoutAttr {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if matches!(self, Self::KTile(0)) {
            return verify_err_noloc!("cute.scaled_layout K tile width must be greater than zero");
        }
        Ok(())
    }
}

/// A byte-alignment promise attached to a tensor view or vector transfer.
///
/// This is a promise, not padding owned by the IR. For example, `16` says
/// the address used by a full tile is safe for one 16-byte transaction.
#[pliron_attr(name = "cute.alignment_bytes", format = "$0")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct CuteAlignmentAttr(pub u64);

impl Verify for CuteAlignmentAttr {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if self.0 == 0 || !self.0.is_power_of_two() {
            return verify_err_noloc!(
                "cute.alignment_bytes must be a positive power of two, got {}",
                self.0
            );
        }
        Ok(())
    }
}

/// Which `mma.sync` operand a warp-cooperative matrix load feeds.
///
/// The role fixes the window shape and the `ldmatrix` variant:
///
/// ```text
/// A: 16x16 window, four 8x8 matrices (.x4), row-major fragment
/// B: 16x8  window, two  8x8 matrices (.x2), transposed in flight
/// ```
#[pliron_attr(name = "cute.matrix_role", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CuteMatrixRoleAttr {
    A,
    B,
}

/// A complete swizzled layout, stored as one typed attribute.
///
/// The quoted payload is intentionally simple and round-trippable:
///
/// ```text
/// "3,4,3;0;elements;(8,32):(32,1)"
///   B M S  off unit      inner layout
/// ```
#[pliron_attr(name = "cute.composed_layout", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct CuteComposedLayoutAttr(pub ComposedLayout);

impl fmt::Display for CuteComposedLayoutAttr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let swizzle = self.0.outer();
        write!(
            f,
            "\"{},{},{};{};{};{}\"",
            swizzle.bits,
            swizzle.base,
            swizzle.shift,
            self.0.offset(),
            self.0.unit(),
            self.0.inner()
        )
    }
}

impl_printable_for_display!(CuteComposedLayoutAttr);

fn parse_composed_layout(text: String) -> Result<CuteComposedLayoutAttr, ParseLayoutError> {
    let mut fields = text.splitn(4, ';');
    let swizzle = fields.next().ok_or(ParseLayoutError)?;
    let offset = fields
        .next()
        .ok_or(ParseLayoutError)?
        .parse::<i64>()
        .map_err(|_| ParseLayoutError)?;
    let unit = match fields.next().ok_or(ParseLayoutError)? {
        "elements" => OffsetUnit::Elements,
        "bytes" => OffsetUnit::Bytes,
        _ => return Err(ParseLayoutError),
    };
    let inner = fields.next().ok_or(ParseLayoutError)?.parse::<Layout>()?;
    let swizzle_fields = swizzle.split(',').map(str::trim).collect::<Vec<_>>();
    let [bits, base, shift] = swizzle_fields.as_slice() else {
        return Err(ParseLayoutError);
    };
    let bits = bits.parse::<u32>().map_err(|_| ParseLayoutError)?;
    let base = base.parse::<u32>().map_err(|_| ParseLayoutError)?;
    let shift = shift.parse::<i32>().map_err(|_| ParseLayoutError)?;
    let outer = Swizzle::try_new(bits, base, shift).ok_or(ParseLayoutError)?;
    ComposedLayout::new(outer, offset, inner, unit)
        .map(CuteComposedLayoutAttr)
        .map_err(|_| ParseLayoutError)
}

impl Parsable for CuteComposedLayoutAttr {
    type Arg = ();
    type Parsed = CuteComposedLayoutAttr;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        quoted_string_parser()
            .and_then(parse_composed_layout)
            .parse_stream(state_stream)
            .into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cute_layout::IntTuple;

    #[test]
    fn composed_layout_text_round_trips_every_addressing_piece() {
        let inner = Layout::new(
            IntTuple::Tuple(vec![IntTuple::Leaf(8), IntTuple::Leaf(32)]),
            IntTuple::Tuple(vec![IntTuple::Leaf(32), IntTuple::Leaf(1)]),
        );
        let composed =
            ComposedLayout::new(Swizzle::new(3, 4, 3), 8, inner, OffsetUnit::Elements).unwrap();
        let attr = CuteComposedLayoutAttr(composed);
        let printed = attr.to_string();
        let reparsed = parse_composed_layout(printed.trim_matches('"').to_string()).unwrap();
        assert_eq!(reparsed, attr);
    }

    #[test]
    fn composed_layout_text_rejects_overlapping_swizzle_fields() {
        // B=4 cannot fit into a shift distance of 3:
        //
        // [ source ]
        //      [ target ]   <- overlap, so XOR is not reversible
        assert!(parse_composed_layout("4,3,3;0;bytes;16:4".to_string()).is_err());
    }

    #[test]
    fn composed_layout_text_preserves_negative_cute_shifts() {
        let attr = parse_composed_layout("2,0,-3;0;bytes;16:4".to_string()).unwrap();
        assert_eq!(attr.0.outer(), Swizzle::new(2, 0, -3));
        assert_eq!(attr.to_string(), "\"2,0,-3;0;bytes;16:4\"");
    }

    #[test]
    fn tensor_structure_rejects_impossible_sizes() {
        let ctx = Context::new();
        assert!(CuteTensorLayoutAttr::Contiguous1D.verify(&ctx).is_ok());
        assert!(CuteTensorLayoutAttr::Zipped1D(4).verify(&ctx).is_ok());
        assert!(CuteTensorLayoutAttr::Tile1D(0).verify(&ctx).is_err());
        assert!(
            CuteTensorLayoutAttr::BlockScaleKMajor(16)
                .verify(&ctx)
                .is_ok()
        );
        assert!(
            CuteTensorLayoutAttr::BlockScaleKMajor(0)
                .verify(&ctx)
                .is_err()
        );
        assert!(CuteScaledLayoutAttr::KTile(64).verify(&ctx).is_ok());
        assert!(CuteScaledLayoutAttr::KTile(0).verify(&ctx).is_err());
        assert!(CuteAlignmentAttr(16).verify(&ctx).is_ok());
        assert!(CuteAlignmentAttr(12).verify(&ctx).is_err());
        assert!(CuteAlignmentAttr(0).verify(&ctx).is_err());
    }
}

/// A CuTe layout (`shape:stride`) as a first-class typed attribute.
#[pliron_attr(name = "cute.layout", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct CuteLayoutAttr(pub Layout);

impl fmt::Display for CuteLayoutAttr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\"{}\"", self.0)
    }
}

impl_printable_for_display!(CuteLayoutAttr);

impl Parsable for CuteLayoutAttr {
    type Arg = ();
    type Parsed = CuteLayoutAttr;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        quoted_string_parser()
            .and_then(|s: String| s.parse::<Layout>().map(CuteLayoutAttr))
            .parse_stream(state_stream)
            .into()
    }
}

/// Register type accumulated by one matrix-multiply atom.
///
/// The first shared-memory MMA slice only needs FP32 accumulation. Keeping it
/// named makes the atom description extensible without hiding the choice in
/// an integer code.
#[pliron_attr(name = "cute.mma_accumulator", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CuteMmaAccumulatorAttr {
    F32,
}

/// What one ordinary MIR aggregate means at a semantic MMA operation.
///
/// B is deliberately absent from this first list. A B partition is a lazy
/// pointer-and-coordinate view, not an eagerly loaded register fragment:
///
/// ```text
/// ScaleStage ── K half ──► ScaleK64
/// A shared tile ─────────► A
/// zero / previous step ──► Accumulator
/// B shared tile ─────────► pointer + capacity + coordinates
/// ```
#[pliron_attr(name = "cute.mma_carrier_kind", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CuteMmaCarrierKindAttr {
    ScaleStage,
    ScaleK64,
    A,
    Accumulator,
}

/// One warp-level matrix-multiply instruction shape.
///
/// Register counts are per lane. They describe the physical MIR aggregates
/// consumed by the selected backend's matrix-instruction lowering:
///
/// ```text
/// A: 4 x u32    B: 2 x u32    C: 4 x f32
///              m16n8k64
/// ```
#[pliron_attr(
    name = "cute.mma_atom",
    format = "`<` $m `,` $n `,` $k `,` $a_format `,` $b_format `,` $scale_format `,` $values_per_scale `,` $accumulator `,` $threads `,` $a_registers `,` $b_registers `,` $accumulator_registers `>`"
)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct CuteMmaAtomAttr {
    pub m: u64,
    pub n: u64,
    pub k: u64,
    pub a_format: CuteTensorFormatAttr,
    pub b_format: CuteTensorFormatAttr,
    pub scale_format: CuteTensorFormatAttr,
    pub values_per_scale: u64,
    pub accumulator: CuteMmaAccumulatorAttr,
    pub threads: u32,
    pub a_registers: u32,
    pub b_registers: u32,
    pub accumulator_registers: u32,
}

impl CuteMmaAtomAttr {
    /// SM120's packed MXFP4 `m16n8k64` atom.
    #[must_use]
    pub const fn mxf4_m16n8k64() -> Self {
        Self {
            m: 16,
            n: 8,
            k: 64,
            a_format: CuteTensorFormatAttr::E2M1,
            b_format: CuteTensorFormatAttr::E2M1,
            scale_format: CuteTensorFormatAttr::UE8M0,
            values_per_scale: 32,
            accumulator: CuteMmaAccumulatorAttr::F32,
            threads: 32,
            a_registers: 4,
            b_registers: 2,
            accumulator_registers: 4,
        }
    }
}

impl Verify for CuteMmaAtomAttr {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if self.m == 0 || self.n == 0 || self.k == 0 {
            return verify_err_noloc!("cute.mma_atom M, N, and K must be greater than zero");
        }
        if self.a_format != CuteTensorFormatAttr::E2M1
            || self.b_format != CuteTensorFormatAttr::E2M1
            || self.scale_format != CuteTensorFormatAttr::UE8M0
        {
            return verify_err_noloc!("cute.mma_atom v0 needs E2M1 A/B values with UE8M0 scales");
        }
        if self.values_per_scale == 0 || !self.k.is_multiple_of(self.values_per_scale) {
            return verify_err_noloc!(
                "cute.mma_atom K must contain a whole positive number of scale groups"
            );
        }
        if self.threads == 0
            || self.a_registers == 0
            || self.b_registers == 0
            || self.accumulator_registers == 0
        {
            return verify_err_noloc!(
                "cute.mma_atom thread and per-lane register counts must be greater than zero"
            );
        }
        Ok(())
    }
}

/// How warp-level MMA atoms cover one CTA output tile.
///
/// `m_ownership` and `n_ownership` map a warp position plus a cell position
/// to the corresponding M or N atom number. The current 128 x 128 plan reads:
///
/// ```text
/// M atoms: (warp_m, cell_m)       (4,2):(1,4)
/// N atoms: ((within_pair,pair),   ((2,4),2):((1,4),2)
///           warp_n)
///
/// 4 warp-M positions x 2 warp-N positions = 8 compute warps
/// one warp owns 2 M atoms x 8 N atoms = 16 accumulator cells
/// ```
///
/// `b_load_group` is the number of B fragments loaded before immediately
/// consuming them. It keeps the performance-critical lazy-B schedule visible.
#[pliron_attr(
    name = "cute.tiled_mma_plan",
    format = "`<` $atom `,` $cta_m `,` $cta_n `,` $cta_k `,` $warp_m `,` $warp_n `,` $b_load_group `,` $shared_layout `,` $m_ownership `,` $n_ownership `>`"
)]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct CuteTiledMmaPlanAttr {
    pub atom: CuteMmaAtomAttr,
    pub cta_m: u64,
    pub cta_n: u64,
    pub cta_k: u64,
    pub warp_m: u32,
    pub warp_n: u32,
    pub b_load_group: u32,
    pub shared_layout: CuteComposedLayoutAttr,
    pub m_ownership: CuteLayoutAttr,
    pub n_ownership: CuteLayoutAttr,
}

impl CuteTiledMmaPlanAttr {
    /// Describe the current 128 x 128 x 128 SM120 MXFP4 tiled MMA.
    #[must_use]
    pub fn mxf4_128x128x128(shared_layout: ComposedLayout) -> Self {
        Self {
            atom: CuteMmaAtomAttr::mxf4_m16n8k64(),
            cta_m: 128,
            cta_n: 128,
            cta_k: 128,
            warp_m: 4,
            warp_n: 2,
            b_load_group: 2,
            shared_layout: CuteComposedLayoutAttr(shared_layout),
            m_ownership: CuteLayoutAttr(
                "(4,2):(1,4)"
                    .parse()
                    .expect("fixed M ownership layout is valid"),
            ),
            n_ownership: CuteLayoutAttr(
                "((2,4),2):((1,4),2)"
                    .parse()
                    .expect("fixed N ownership layout is valid"),
            ),
        }
    }

    /// Number of compute warps in this plan.
    #[must_use]
    pub fn compute_warps(&self) -> Option<u64> {
        u64::from(self.warp_m).checked_mul(u64::from(self.warp_n))
    }

    /// Number of M atoms owned by one warp.
    #[must_use]
    pub fn m_atoms_per_warp(&self) -> Option<u64> {
        self.cta_m
            .checked_div(self.atom.m)?
            .checked_div(u64::from(self.warp_m))
    }

    /// Number of N atoms owned by one warp.
    #[must_use]
    pub fn n_atoms_per_warp(&self) -> Option<u64> {
        self.cta_n
            .checked_div(self.atom.n)?
            .checked_div(u64::from(self.warp_n))
    }

    /// Packed scale words held by one lane for one K=128 stage.
    #[must_use]
    pub fn scale_words_per_lane(&self) -> Option<u64> {
        self.m_atoms_per_warp()?
            .checked_add(self.n_atoms_per_warp()?)
    }

    /// A-fragment u32 registers held by one lane for one K=64 step.
    #[must_use]
    pub fn a_registers_per_lane(&self) -> Option<u64> {
        self.m_atoms_per_warp()?
            .checked_mul(u64::from(self.atom.a_registers))
    }

    /// FP32 accumulator registers held by one lane for the complete warp tile.
    #[must_use]
    pub fn accumulator_registers_per_lane(&self) -> Option<u64> {
        self.m_atoms_per_warp()?
            .checked_mul(self.n_atoms_per_warp()?)?
            .checked_mul(u64::from(self.atom.accumulator_registers))
    }
}

fn layout_is_dense_permutation(layout: &Layout) -> bool {
    let Some(size) = layout.checked_size() else {
        return false;
    };
    let mut offsets = (0..size)
        .map(|coordinate| layout.checked_call(&cute_layout::IntTuple::Leaf(coordinate)))
        .collect::<Option<Vec<_>>>();
    let Some(ref mut offsets) = offsets else {
        return false;
    };
    offsets.sort_unstable();
    offsets.iter().copied().eq(0..size)
}

impl Verify for CuteTiledMmaPlanAttr {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        self.atom.verify(ctx)?;
        if self.cta_m == 0 || self.cta_n == 0 || self.cta_k == 0 {
            return verify_err_noloc!(
                "cute.tiled_mma_plan CTA M, N, and K must be greater than zero"
            );
        }
        if !self.cta_m.is_multiple_of(self.atom.m)
            || !self.cta_n.is_multiple_of(self.atom.n)
            || !self.cta_k.is_multiple_of(self.atom.k)
        {
            return verify_err_noloc!("cute.tiled_mma_plan CTA tile must contain whole MMA atoms");
        }
        if self.warp_m == 0 || self.warp_n == 0 {
            return verify_err_noloc!(
                "cute.tiled_mma_plan warp grid sizes must be greater than zero"
            );
        }
        let m_atoms = self.cta_m / self.atom.m;
        let n_atoms = self.cta_n / self.atom.n;
        if !m_atoms.is_multiple_of(u64::from(self.warp_m))
            || !n_atoms.is_multiple_of(u64::from(self.warp_n))
        {
            return verify_err_noloc!(
                "cute.tiled_mma_plan atom grid must divide evenly across its warp grid"
            );
        }
        let Some(compute_warps) = self.compute_warps() else {
            return verify_err_noloc!("cute.tiled_mma_plan compute-warp count overflowed");
        };
        if compute_warps == 0 || compute_warps > 32 {
            return verify_err_noloc!(
                "cute.tiled_mma_plan must use between 1 and 32 compute warps"
            );
        }
        let Some(n_per_warp) = self.n_atoms_per_warp() else {
            return verify_err_noloc!("cute.tiled_mma_plan N ownership is not divisible");
        };
        if self.b_load_group == 0 || n_per_warp % u64::from(self.b_load_group) != 0 {
            return verify_err_noloc!(
                "cute.tiled_mma_plan B load group must divide one warp's N atoms"
            );
        }
        if self.shared_layout.0.unit() != OffsetUnit::Elements || self.shared_layout.0.offset() != 0
        {
            return verify_err_noloc!(
                "cute.tiled_mma_plan shared placement must use zero-based element offsets"
            );
        }
        let expected_shared_elements = self
            .cta_m
            .checked_mul(self.cta_k)
            .and_then(|values| values.checked_div(4));
        if expected_shared_elements.and_then(|value| i64::try_from(value).ok())
            != self.shared_layout.0.inner().checked_size()
        {
            return verify_err_noloc!(
                "cute.tiled_mma_plan shared placement must hold four packed E2M1 values per f16 carrier"
            );
        }
        if self.m_ownership.0.checked_size() != i64::try_from(m_atoms).ok()
            || self.n_ownership.0.checked_size() != i64::try_from(n_atoms).ok()
            || !layout_is_dense_permutation(&self.m_ownership.0)
            || !layout_is_dense_permutation(&self.n_ownership.0)
        {
            return verify_err_noloc!(
                "cute.tiled_mma_plan ownership layouts must cover every M/N atom exactly once"
            );
        }
        if self.m_ownership.0.shape.leaves().first().copied() != Some(i64::from(self.warp_m))
            || self.n_ownership.0.shape.leaves().last().copied() != Some(i64::from(self.warp_n))
        {
            return verify_err_noloc!(
                "cute.tiled_mma_plan ownership layouts must expose warp-M first and warp-N last"
            );
        }
        Ok(())
    }
}

/// How many result buffers the asynchronous TMA store queue may reuse.
///
/// The value has no runtime storage. It only explains the wait count used by
/// the three store-pipeline operations:
///
/// ```text
/// acquire  waits for at most stages - 1 old readers
/// commit   closes the stores issued since the previous commit
/// tail     waits for zero old readers
/// ```
#[pliron_attr(name = "cute.tma_store_pipeline", format = "`<` $stages `>`")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct CuteTmaStorePipelineAttr {
    pub stages: u32,
}

impl CuteTmaStorePipelineAttr {
    #[must_use]
    pub const fn new(stages: u32) -> Self {
        Self { stages }
    }

    /// Number passed to `cp.async.bulk.wait_group.read` before reuse.
    #[must_use]
    pub const fn max_pending(self) -> Option<u32> {
        self.stages.checked_sub(1)
    }
}

impl Verify for CuteTmaStorePipelineAttr {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if self.stages == 0 {
            return verify_err_noloc!("cute.tma_store_pipeline needs at least one stage");
        }
        Ok(())
    }
}

/// One named CTA barrier used by a contiguous group of warps.
///
/// The current epilogue uses warps 0 through 7 and leaves warp 8 out:
///
/// ```text
/// CTA warps       0 1 2 3 4 5 6 7 | 8
/// barrier ID 2    [--- 256 threads ---] | producer
/// ```
#[pliron_attr(
    name = "cute.counted_cta_barrier",
    format = "`<` $barrier_id `,` $first_warp `,` $warp_count `,` $cta_warps `,` $lanes_per_warp `>`"
)]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct CuteCountedCtaBarrierAttr {
    pub barrier_id: u32,
    pub first_warp: u32,
    pub warp_count: u32,
    pub cta_warps: u32,
    pub lanes_per_warp: u32,
}

impl CuteCountedCtaBarrierAttr {
    #[must_use]
    pub const fn new(
        barrier_id: u32,
        first_warp: u32,
        warp_count: u32,
        cta_warps: u32,
        lanes_per_warp: u32,
    ) -> Self {
        Self {
            barrier_id,
            first_warp,
            warp_count,
            cta_warps,
            lanes_per_warp,
        }
    }

    /// Threads that must arrive at the counted barrier.
    #[must_use]
    pub const fn participant_threads(self) -> Option<u32> {
        self.warp_count.checked_mul(self.lanes_per_warp)
    }

    /// Warps in the CTA that deliberately do not arrive.
    #[must_use]
    pub const fn excluded_warps(self) -> Option<u32> {
        self.cta_warps.checked_sub(self.warp_count)
    }
}

impl Verify for CuteCountedCtaBarrierAttr {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if self.barrier_id >= 16 {
            return verify_err_noloc!("cute.counted_cta_barrier ID must be between 0 and 15");
        }
        if self.warp_count == 0 || self.cta_warps == 0 || self.lanes_per_warp == 0 {
            return verify_err_noloc!(
                "cute.counted_cta_barrier warp and lane counts must be greater than zero"
            );
        }
        let Some(end_warp) = self.first_warp.checked_add(self.warp_count) else {
            return verify_err_noloc!("cute.counted_cta_barrier warp range overflowed");
        };
        if end_warp > self.cta_warps {
            return verify_err_noloc!(
                "cute.counted_cta_barrier participating warps must lie inside the CTA"
            );
        }
        let Some(threads) = self.participant_threads() else {
            return verify_err_noloc!("cute.counted_cta_barrier thread count overflowed");
        };
        let Some(cta_threads) = self.cta_warps.checked_mul(self.lanes_per_warp) else {
            return verify_err_noloc!("cute.counted_cta_barrier CTA thread count overflowed");
        };
        if threads == 0 || threads > 1024 || cta_threads > 1024 {
            return verify_err_noloc!(
                "cute.counted_cta_barrier cannot describe more than 1024 CTA threads"
            );
        }
        Ok(())
    }
}

/// Why the compute warps meet at the epilogue barrier.
#[pliron_attr(name = "cute.epilogue_sync_phase", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum CuteEpilogueSyncPhaseAttr {
    /// The previous TMA reader is done, so writers may reuse shared memory.
    Reusable,
    /// Publish generic-proxy writes, then release TMA to read shared memory.
    ReadyForTma,
}

/// Left or right 128x64 half of the current 128x128 result tile.
#[pliron_attr(name = "cute.epilogue_half_index", format = "$0")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct CuteEpilogueHalfAttr(pub u32);

impl Verify for CuteEpilogueHalfAttr {
    fn verify(&self, _ctx: &Context) -> Result<(), Error> {
        if self.0 >= 2 {
            return verify_err_noloc!("cute.epilogue_half_index v0 must be 0 or 1");
        }
        Ok(())
    }
}

/// How one warp's FP32 accumulator reaches a shared FP16 result tile.
///
/// ```text
/// 2 x 8 accumulator cells per warp
///              │
///              ▼
/// logical 128x128 f16 tile
/// ┌────── 128x64 ──────┬────── 128x64 ──────┐
/// │ one B128 TMA store │ one B128 TMA store │
/// └────────────────────┴────────────────────┘
/// ```
#[pliron_attr(
    name = "cute.epilogue_plan",
    format = "`<` $tiled_mma `,` $half_layout `,` $halves `,` $base_alignment `>`"
)]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct CuteEpiloguePlanAttr {
    pub tiled_mma: CuteTiledMmaPlanAttr,
    pub half_layout: CuteComposedLayoutAttr,
    pub halves: u32,
    pub base_alignment: CuteAlignmentAttr,
}

impl CuteEpiloguePlanAttr {
    /// Current SM120 two-half FP16 epilogue for the 128x128 MXFP4 plan.
    #[must_use]
    pub fn sm120_mxf4_128x128(tiled_mma: CuteTiledMmaPlanAttr) -> Self {
        let inner: Layout = "(128,64):(64,1)"
            .parse()
            .expect("fixed epilogue half layout is valid");
        let half_layout =
            ComposedLayout::new(Swizzle::new(3, 3, 3), 0, inner, OffsetUnit::Elements)
                .expect("fixed epilogue swizzle is valid");
        Self {
            tiled_mma,
            half_layout: CuteComposedLayoutAttr(half_layout),
            halves: 2,
            base_alignment: CuteAlignmentAttr(1024),
        }
    }

    #[must_use]
    pub fn half_elements(&self) -> Option<u64> {
        u64::try_from(self.half_layout.0.inner().checked_size()?).ok()
    }

    #[must_use]
    pub fn full_elements(&self) -> Option<u64> {
        self.half_elements()?.checked_mul(u64::from(self.halves))
    }

    #[must_use]
    pub fn half_bytes(&self, element_bytes: u64) -> Option<u64> {
        self.half_elements()?.checked_mul(element_bytes)
    }

    #[must_use]
    pub fn full_bytes(&self, element_bytes: u64) -> Option<u64> {
        self.full_elements()?.checked_mul(element_bytes)
    }
}

impl Verify for CuteEpiloguePlanAttr {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        self.tiled_mma.verify(ctx)?;
        self.base_alignment.verify(ctx)?;

        let data_inner: Layout = "(128,32):(32,1)"
            .parse()
            .expect("fixed SM120 MMA shared layout is valid");
        let data_layout =
            ComposedLayout::new(Swizzle::new(2, 3, 3), 0, data_inner, OffsetUnit::Elements)
                .expect("fixed SM120 MMA shared swizzle is valid");
        let expected_mma = CuteTiledMmaPlanAttr::mxf4_128x128x128(data_layout);
        if self.tiled_mma != expected_mma {
            return verify_err_noloc!(
                "cute.epilogue_plan v0 needs the 128x128x128 SM120 MXFP4 tiled-MMA plan"
            );
        }
        let expected = Self::sm120_mxf4_128x128(self.tiled_mma.clone());
        if self.half_layout != expected.half_layout
            || self.halves != expected.halves
            || self.base_alignment != expected.base_alignment
        {
            return verify_err_noloc!(
                "cute.epilogue_plan v0 needs two B128-swizzled 128x64 f16 halves aligned to 1024 bytes"
            );
        }
        let modes = self.half_layout.0.inner().modes();
        if modes.len() != 2
            || modes[0].checked_size() != Some(128)
            || modes[1].checked_size() != Some(64)
            || self.full_elements() != Some(128 * 128)
        {
            return verify_err_noloc!(
                "cute.epilogue_plan halves must cover one 128x128 result tile"
            );
        }
        if cute_layout::tma_phase_alignment_bytes(&self.half_layout.0, 2) != Ok(1024) {
            return verify_err_noloc!(
                "cute.epilogue_plan half layout must use the 1024-byte B128 phase alignment"
            );
        }
        if self.tiled_mma.accumulator_registers_per_lane() != Some(64) {
            return verify_err_noloc!(
                "cute.epilogue_plan v0 needs 64 FP32 accumulator registers per lane"
            );
        }
        Ok(())
    }
}
