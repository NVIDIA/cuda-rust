/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
//! Warp-level FP16/BF16 and SM120 MXFP4 instructions with role-specific registers.
//! Addresses, lane mapping, synchronization and issue order remain explicit.
use super::memory::SharedAddress;

/// Map one lane's accumulator register to its row/column in an m16n8 tile.
///
/// Registers 0/1 are adjacent columns in the lower row; registers 2/3 hold
/// the row eight positions later. As in the canonical mapping, lane and
/// register indices are reduced modulo 32 and 4 respectively.
#[must_use]
#[inline(always)]
pub const fn acc_coords(lane: usize, register: usize) -> (usize, usize) {
    cuda_device::mma_frag::acc_coords(lane, register)
}

/// Expand a body once per listed index, with a const-generic-compatible index.
///
/// Each expansion has its own block scope and executes in list order. Nest
/// invocations for multiple axes. This is source expansion, not a runtime loop
/// or a compiler unroll hint; it does not imply a particular register allocation.
///
/// ```
/// use ctm_rs::mma::{self, AccumulatorTile16x8};
/// let tile = AccumulatorTile16x8::<2, 8>::zero();
/// mma::static_for!(ROW in [0, 1] {
///     mma::static_for!(COL in [0, 1, 2, 3, 4, 5, 6, 7] {
///         assert_eq!(tile.fragment::<ROW, COL>().registers(), [0.0; 4]);
///     });
/// });
/// ```
#[macro_export]
macro_rules! static_for {
    ($index:ident in [$($value:literal),* $(,)?] $body:block) => {
        $({
            /// Compile-time coordinate for this expansion in the requested index order.
            const $index: usize = $value;
            $body
        })*
    };
}
pub use crate::static_for;

/// Define an operand-role register carrier and its by-value accessors.
macro_rules! registers {
    ($name:ident, $element:ty, $count:literal, $description:literal) => {
        #[doc = $description]
        #[repr(transparent)]
        #[derive(Clone, Copy)]
        pub struct $name(
            #[doc = "Per-lane register values in the instruction's operand order."]
            [$element; $count],
        );
        impl $name {
            /// Adopt registers without changing their bits or lane ownership.
            #[inline(always)]
            pub const fn from_registers(registers: [$element; $count]) -> Self {
                Self(registers)
            }
            /// Return the registers by value in their original instruction order.
            #[inline(always)]
            pub const fn registers(self) -> [$element; $count] {
                self.0
            }
        }
    };
}
registers!(
    F16A,
    u32,
    4,
    "One lane's packed FP16 A fragment for m16n8k16, in row-layout PTX register order."
);
registers!(
    F16B,
    u32,
    2,
    "One lane's packed FP16 B fragment for m16n8k16, in column-layout PTX register order."
);
registers!(
    Bf16A,
    u32,
    4,
    "One lane's packed BF16 A fragment for m16n8k16, in row-layout PTX register order."
);
registers!(
    Bf16B,
    u32,
    2,
    "One lane's packed BF16 B fragment for m16n8k16, in column-layout PTX register order."
);
registers!(
    Mxf4A,
    u32,
    4,
    "One lane's packed E2M1 A fragment for m16n8k64, in PTX register order."
);
registers!(
    Mxf4B,
    u32,
    2,
    "One lane's packed E2M1 B fragment for m16n8k64, in PTX register order."
);
registers!(
    Accumulator16x8,
    f32,
    4,
    "One lane's four FP32 accumulators for a warp-wide 16x8 output."
);
impl Accumulator16x8 {
    /// Initialize this lane’s four accumulator registers to FP32 zero.
    #[inline(always)]
    pub const fn zero() -> Self {
        Self([0.0; 4])
    }
}

/// Issue one m16n8k16 row/column FP16 MMA, accumulating in FP32.
///
/// The four A registers and two B registers each pack two FP16 values. The
/// four accumulator registers retain PTX result order. This forwards exactly
/// `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`, without loads,
/// synchronization, election or rearrangement of registers.
///
/// # Safety
/// All 32 lanes must execute uniformly on SM80 or newer. A, B and C must
/// follow the PTX lane/register mappings for the same warp-wide product.
#[must_use = "use the returned accumulator; mma does not modify c in place"]
#[inline(always)]
pub unsafe fn mma_f16(c: Accumulator16x8, a: F16A, b: F16B) -> Accumulator16x8 {
    Accumulator16x8(unsafe { cuda_device::wmma::mma_m16n8k16_f32_f16(c.0, a.0, b.0) })
}

/// Issue one m16n8k16 row/column BF16 MMA, accumulating in FP32.
///
/// Each A/B register packs two BF16 values, with the first in its low sixteen
/// bits. Register and lane order follow
/// [`cuda_device::wmma::mma_m16n8k16_f32_bf16`]. The four FP32 result registers
/// retain PTX accumulator order. No conversion, load, synchronization or
/// rearrangement is inserted.
///
/// # Safety
/// All 32 lanes must execute uniformly on SM80 or newer. A, B and C must
/// follow the PTX lane/register mappings for the same warp-wide product.
#[must_use = "use the returned accumulator; mma does not modify c in place"]
#[inline(always)]
pub unsafe fn mma_bf16(c: Accumulator16x8, a: Bf16A, b: Bf16B) -> Accumulator16x8 {
    Accumulator16x8(unsafe { cuda_device::wmma::mma_m16n8k16_f32_bf16(c.0, a.0, b.0) })
}

/// One lane's rectangular collection of warp-wide 16x8 accumulator fragments.
///
/// `ROWS` and `COLS` count fragments, not scalar elements. Physical placement is
/// the caller's responsibility: adjacent indices need not be adjacent in memory.
/// Access is const-indexed so callers can express a static instruction schedule
/// without runtime indexing into the register collection.
#[repr(transparent)]
pub struct AccumulatorTile16x8<const ROWS: usize, const COLS: usize> {
    /// Per-lane accumulator fragments indexed by logical fragment row and column.
    fragments: [[Accumulator16x8; COLS]; ROWS],
}

impl<const ROWS: usize, const COLS: usize> AccumulatorTile16x8<ROWS, COLS> {
    /// Initialize every fragment's four FP32 accumulators to zero.
    #[inline(always)]
    pub const fn zero() -> Self {
        Self {
            fragments: [[Accumulator16x8::zero(); COLS]; ROWS],
        }
    }

    /// Read one fragment; an out-of-range coordinate fails at compile time.
    #[inline(always)]
    pub const fn fragment<const ROW: usize, const COL: usize>(&self) -> Accumulator16x8 {
        const {
            assert!(
                ROW < ROWS && COL < COLS,
                "accumulator fragment out of bounds"
            )
        }
        self.fragments[ROW][COL]
    }

    /// Mutably borrow one fragment at a compile-time coordinate.
    #[inline(always)]
    pub fn fragment_mut<const ROW: usize, const COL: usize>(&mut self) -> &mut Accumulator16x8 {
        const {
            assert!(
                ROW < ROWS && COL < COLS,
                "accumulator fragment out of bounds"
            )
        }
        &mut self.fragments[ROW][COL]
    }
}

impl<const COLS: usize> AccumulatorTile16x8<2, COLS> {
    /// Accumulate one B pair into both fragment rows using exactly four MMAs.
    ///
    /// Issue order is `(row 0, left)`, `(row 0, right)`, `(row 1, left)`, then
    /// `(row 1, right)`, where `left = 2 * PAIR`. No loads or synchronization
    /// are inserted. The caller can load the next B pair after this operation.
    ///
    /// # Safety
    /// The collective, architecture and register-mapping requirements of
    /// [`mma_mxf4`] apply to every fragment and scale pair.
    #[inline(always)]
    pub unsafe fn mma_mxf4_pair<const PAIR: usize>(
        &mut self,
        a: [Mxf4A; 2],
        b: [Mxf4B; 2],
        scale_a: [ScalePairA; 2],
        scale_b: [ScalePairB; 2],
    ) {
        const { assert!(PAIR < COLS / 2, "accumulator column pair out of bounds") }
        static_for!(ROW in [0, 1] {
            static_for!(COL in [0, 1] {
                let c = &mut self.fragments[ROW][const { 2 * PAIR + COL }];
                *c = unsafe { mma_mxf4(*c, a[ROW], b[COL], scale_a[ROW], scale_b[COL]) };
            });
        });
    }
}

/// Define role-specific packed scale pairs with K32 scales in low/high bytes.
macro_rules! scales {
    ($name:ident, $role:literal) => {
        #[doc = $role]
        #[repr(transparent)]
        #[derive(Clone, Copy)]
        pub struct $name(#[doc = "Two UE8M0 scales, low byte first along K."] u16);
        impl $name {
            /// Low byte scales K[0..32], high byte scales K[32..64].
            #[inline(always)]
            pub const fn from_bits(bits: u16) -> Self {
                Self(bits)
            }
            /// Return both packed scale bytes unchanged.
            #[inline(always)]
            pub const fn bits(self) -> u16 {
                self.0
            }
        }
    };
}
scales!(ScalePairA, "Packed UE8M0 scales for operand A.");
scales!(ScalePairB, "Packed UE8M0 scales for operand B.");

/// Load four A registers using one row-layout ldmatrix.x4 instruction.
///
/// # Safety
/// All 32 lanes must participate with live, aligned shared row addresses in the
/// PTX m8n8.x4 mapping. Order writes/reads before and after this weak memory op.
#[inline(always)]
pub unsafe fn load_mxf4_a(address: SharedAddress<'_, u32>) -> Mxf4A {
    Mxf4A(unsafe { cuda_device::wmma::ldmatrix_x4(address.as_raw()) })
}
/// Load one B pair with ldmatrix.x4; separate the even/odd row submatrices.
///
/// # Safety
/// Same collective/address requirements as [`load_mxf4_a`]. Addresses must
/// select two neighboring N8 fragments of an E2M1 K64 tile.
#[inline(always)]
pub unsafe fn load_mxf4_b_pair(address: SharedAddress<'_, u32>) -> (Mxf4B, Mxf4B) {
    let r = unsafe { cuda_device::wmma::ldmatrix_x4(address.as_raw()) };
    (Mxf4B([r[0], r[2]]), Mxf4B([r[1], r[3]]))
}
/// Issue one SM120 m16n8k64 E2M1/UE8M0 block-scaled MMA with selectors zero.
/// No loads, synchronization, election or completion waits are inserted.
///
/// # Safety
/// All 32 lanes must execute uniformly on SM120a. Fragments and per-lane scale
/// providers must follow the PTX register mapping for the same K64 tile.
///
/// ```compile_fail
/// use ctm_rs::mma::*;
/// unsafe {
///     mma_mxf4(Accumulator16x8::zero(), Mxf4A::from_registers([0;4]),
///         Mxf4B::from_registers([0;2]), ScalePairB::from_bits(0), ScalePairA::from_bits(0));
/// }
/// ```
#[inline(always)]
pub unsafe fn mma_mxf4(
    c: Accumulator16x8,
    a: Mxf4A,
    b: Mxf4B,
    scale_a: ScalePairA,
    scale_b: ScalePairB,
) -> Accumulator16x8 {
    Accumulator16x8(unsafe {
        cuda_device::wmma::mma_m16n8k64_mxf4_f32(
            c.0,
            a.0,
            b.0,
            scale_a.0 as u32,
            0,
            0,
            scale_b.0 as u32,
            0,
            0,
        )
    })
}
/// Convert four accumulators to FP16 and issue one row-layout stmatrix.x2.
///
/// # Safety
/// All 32 lanes must participate. Shared row addresses must satisfy PTX x2
/// mapping and alignment, and refer to live output storage. Explicitly publish
/// the writes to the async proxy before any dependent TMA transfer.
#[inline(always)]
pub unsafe fn store_f16(address: SharedAddress<'_, u8>, c: Accumulator16x8) {
    let top = crate::convert::cvt_f16x2_f32(c.0[0], c.0[1]);
    let bottom = crate::convert::cvt_f16x2_f32(c.0[2], c.0[3]);
    unsafe { cuda_device::tcgen05::stmatrix_m8n8_x2(address.as_raw(), top, bottom) }
}
