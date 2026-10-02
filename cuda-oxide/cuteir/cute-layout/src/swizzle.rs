/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Reorder offsets by XOR-ing some of their bits.
//!
//! A layout first turns a logical coordinate such as `(row, column)` into a
//! **tile-relative offset**. An offset counts from the start of the tile; it
//! is not a pointer. Depending on the layout, one step may mean one element
//! or one byte. A [`Swizzle`] changes that offset before the allocation's base
//! pointer is added:
//!
//! ```text
//! (row, column) ── layout ──> plain offset ── swizzle ──> physical offset
//!                                                           │
//! allocation base pointer + physical offset ────────────────┘
//! ```
//!
//! Shared memory is split into hardware banks. A plain row-major layout can
//! make many threads ask the same bank for data at once. A swizzle permutes
//! chunks of a tile so those requests spread across banks. It changes where
//! values are stored; it does not copy data, fill memory, or change the
//! number of values.
//!
//! ## Reading `S<B,M,S>`
//!
//! The offset is treated as a binary number, with bit 0 on the right:
//!
//! ```text
//! positive S:
//!
//! higher bits                                                    lower bits
//! [ source: B bits ][ gap: S-B bits ][ target: B bits ][ unchanged: M bits ]
//!          └──────────── shift right S, then XOR ─────▲
//! ```
//!
//! - `B` is the width of the two bit fields. It is not necessarily the
//!   number of row bits; which bits describe rows depends on the preceding
//!   layout.
//! - `M` is the number of lowest bits that the swizzle never changes.
//! - `S` is the signed distance from the target field to the source field.
//!   Positive `S` reads a higher field and moves it right. Negative `S` reads
//!   a lower field and moves it left, reversing the source and target positions
//!   in the diagram.
//!
//! The shifted source is XOR-ed with the existing target. XOR toggles target
//! bits where the source has a 1; it does not replace the target field.
//!
//! For example, `S<2,0,3>` copies bits 4..3 down to bits 1..0 and XORs them:
//!
//! ```text
//! input  0b0_01_0_00 = 8
//!              01  ── move right by 3 ──> 01
//! output 0b0_01_0_01 = 9
//! ```
//!
//! The source and target fields must not overlap, which is why `|S| >= B`.
//! Both fields must also fit in the supported low 32-bit swizzle window.
//! Higher offset bits pass through unchanged.
//!
//! ## What the protected bits guarantee
//!
//! If offsets count bytes, `M = 4` leaves the lowest four address-offset bits
//! unchanged. Each **aligned** 16-byte chunk therefore remains one aligned,
//! contiguous 16-byte chunk after swizzling, although that whole chunk may
//! move elsewhere:
//!
//! ```text
//! before: chunk number | byte within chunk
//!                         bits 3..0
//! after:  new chunk     | same bits 3..0
//! ```
//!
//! In general, a swizzle alone preserves aligned chunks of `2^M` input
//! units. If the input counts elements, that means `2^M` elements, not bytes.
//! This is only one vectorization condition: the inner layout, composed
//! offset, and real pointer must provide compatible alignment too.

use core::fmt;

/// A reversible permutation of a tile-relative offset's bits.
///
/// `Swizzle { bits: B, base: M, shift: S }` represents CuTe's `S<B,M,S>`.
/// It changes only the offset supplied to [`Swizzle::apply`]; it does not own
/// memory or move any values by itself. See the module-level bit diagram for
/// the meaning of all three fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Swizzle {
    /// `B`: number of bits copied from the source field and XOR-ed into the
    /// target field.
    pub bits: u32,
    /// `M`: number of lowest offset bits that are never changed.
    ///
    /// When offsets count bytes, this preserves aligned chunks of `2^M`
    /// bytes. When offsets count elements, it preserves `2^M` elements.
    pub base: u32,
    /// `S`: signed distance from target field to source field.
    ///
    /// A positive value moves a higher source field right. A negative value
    /// moves a lower source field left.
    pub shift: i32,
}

impl Swizzle {
    /// Highest bit range that the swizzle's fields may occupy.
    ///
    /// This limits the source and target field positions, not the input value's
    /// width: [`Swizzle::apply`] preserves any higher bits. The allocation's
    /// absolute base address is never part of the swizzled offset.
    pub const BIT_WIDTH: u32 = 32;

    /// Build `S<B,M,S>` and panic if its bit fields are impossible.
    ///
    /// A valid swizzle has two separate `B`-bit fields, and both fields fit in
    /// the supported low 32-bit swizzle window:
    ///
    /// ```text
    /// source and target do not overlap:  |S| >= B
    /// highest used bit fits:             M + B + |S| <= 32
    /// ```
    pub const fn new(bits: u32, base: u32, shift: i32) -> Self {
        assert!(
            Self::parameters_are_valid(bits, base, shift),
            "invalid CuTe swizzle"
        );
        Swizzle { bits, base, shift }
    }

    /// Build `S<B,M,S>`, returning `None` when its fields overlap or fall
    /// outside the supported low 32-bit swizzle window.
    ///
    /// Compiler passes use this form for values decoded from Rust types or the
    /// compiler's intermediate representation (IR), because malformed input
    /// must produce a diagnostic rather than panic the compiler.
    pub const fn try_new(bits: u32, base: u32, shift: i32) -> Option<Self> {
        if Self::parameters_are_valid(bits, base, shift) {
            Some(Swizzle { bits, base, shift })
        } else {
            None
        }
    }

    /// Return whether `S<B,M,S>` has disjoint fields inside the supported low
    /// 32-bit swizzle window.
    ///
    /// `bits`, `base`, and `shift` mean `B`, `M`, and `S`, respectively.
    pub const fn parameters_are_valid(bits: u32, base: u32, shift: i32) -> bool {
        let distance = shift.unsigned_abs();
        if distance < bits {
            return false;
        }
        let Some(base_and_bits) = base.checked_add(bits) else {
            return false;
        };
        let Some(used_bits) = base_and_bits.checked_add(distance) else {
            return false;
        };
        used_bits <= Self::BIT_WIDTH
    }

    /// Return whether this swizzle has disjoint fields inside the supported
    /// low 32-bit swizzle window.
    ///
    /// The fields are public so compiler code can decode them directly. That
    /// also means a caller can create an invalid struct literal, so an input
    /// boundary must call this method before using such a value.
    pub const fn is_valid(&self) -> bool {
        Self::parameters_are_valid(self.bits, self.base, self.shift)
    }

    /// A swizzle that leaves every offset unchanged.
    ///
    /// `B = 0` makes both bit fields empty. The `M` and `S` values are then
    /// harmless placeholders chosen to match CuTe's canonical spelling.
    pub const IDENTITY: Swizzle = Swizzle::new(0, 4, 3);

    /// Return a mask with 1s at every source-bit position.
    ///
    /// A mask selects bits with bitwise AND. For `S<2,0,3>`, the positive
    /// shift places the source in bits 4..3:
    ///
    /// ```text
    /// y_mask = 0b11000
    /// offset & y_mask keeps only those two source bits
    /// ```
    pub fn y_mask(&self) -> i64 {
        debug_assert!(self.is_valid());
        let mask = (1i64 << self.bits) - 1;
        mask << (self.base as i64 + self.shift.max(0) as i64)
    }

    /// Return a mask with 1s at every target-bit position.
    ///
    /// These are the positions that may be toggled after the source field is
    /// shifted. For `S<2,0,3>`, the target is bits 1..0, so the result is
    /// `0b11`.
    pub fn z_mask(&self) -> i64 {
        debug_assert!(self.is_valid());
        let mask = (1i64 << self.bits) - 1;
        mask << (self.base as i64 + (-self.shift).max(0) as i64)
    }

    /// Map one plain tile-relative offset to its physical offset.
    ///
    /// The method is unit-agnostic: if `x` counts bytes, the result counts
    /// bytes; if `x` counts elements, the result counts elements. Do not add
    /// an absolute pointer before calling this method: pointer-address bits
    /// covered by the source mask would change the XOR result.
    pub fn apply(&self, x: i64) -> i64 {
        debug_assert!(self.is_valid());
        let masked = x & self.y_mask();
        let moved = if self.shift >= 0 {
            masked >> self.shift
        } else {
            masked << (-self.shift)
        };
        x ^ moved
    }

    /// Return whether applying this swizzle twice restores the original
    /// offset.
    ///
    /// This is called an *involution*. It holds when source and target fields
    /// do not overlap: the first application XORs the source into the target,
    /// and the second application XORs the same bits back out.
    pub fn is_involution(&self) -> bool {
        let y = self.y_mask();
        let z = if self.shift >= 0 {
            y >> self.shift
        } else {
            y << (-self.shift)
        };
        y & z == 0
    }

    /// Alignment factor encoded by this swizzle's unchanged low bits.
    ///
    /// `M` unchanged low bits keep every aligned `2^M`-unit chunk contiguous:
    ///
    /// ```text
    /// S<3,4,3> with byte offsets
    ///                 └── 2^4 = 16-byte chunks remain intact
    /// ```
    ///
    /// The answer uses the input offset's unit. It does not by itself prove a
    /// legal vector access: the layout, composed offset, and allocation base
    /// must satisfy the same alignment. For [`Swizzle::IDENTITY`], `M = 4` is
    /// only a canonical placeholder; the identity mapping itself splits no
    /// chunk of any size.
    pub const fn max_alignment(&self) -> i64 {
        debug_assert!(self.is_valid());
        1i64 << self.base
    }

    /// Translate CuTe's positive relative shift into `cuda-device`'s absolute
    /// source-bit position.
    ///
    /// ```text
    /// CuTe S<3,3,3>
    ///
    /// bit 8..6       bit 5..3       bit 2..0
    /// [ source ]     [ target ]     [ unchanged ]
    ///
    /// relative distance S = 3
    /// absolute first source bit = M + S = 3 + 3 = 6
    /// ```
    ///
    /// CuTe stores the distance between the fields. `cuda-device` stores the
    /// first source-bit number. `cuda-device` can only move a higher source
    /// field down, so a negative CuTe shift has no representation and returns
    /// `None`. Invalid swizzles also return `None`.
    pub const fn cuda_device_absolute_source_bit(&self) -> Option<u32> {
        if !self.is_valid() || self.shift < 0 {
            return None;
        }
        self.base.checked_add(self.shift as u32)
    }
}

impl fmt::Display for Swizzle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "S<{},{},{}>", self.bits, self.base, self.shift)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hand_checked_values() {
        // With B=2 and M=0, the target is bits 1..0. A positive S=3
        // selects source bits 4..3 and moves them down before the XOR.
        let s = Swizzle::new(2, 0, 3);
        assert_eq!(s.apply(0b01000), 0b01001);
        assert_eq!(s.apply(0b10000), 0b10010);
        // Inputs with no source bits set pass through unchanged.
        assert_eq!(s.apply(0b00111), 0b00111);

        // A negative S reverses the direction: bits 1..0 are the source and
        // move left into bits 4..3.
        let n = Swizzle::new(2, 0, -3);
        assert_eq!(n.y_mask(), 0b11);
        assert_eq!(n.apply(0b00001), 0b01001);
    }

    #[test]
    fn validity_matches_cutegen_boundaries() {
        assert!(Swizzle::parameters_are_valid(3, 3, 3));
        assert!(Swizzle::parameters_are_valid(0, 32, 0));
        // Four-bit fields only three bits apart overlap.
        assert!(!Swizzle::parameters_are_valid(4, 3, 3));
        // M + B + |S| = 33, one bit beyond the supported swizzle window.
        assert!(!Swizzle::parameters_are_valid(3, 27, 3));
        assert!(!Swizzle::parameters_are_valid(1, 0, i32::MIN));
        assert!(Swizzle::try_new(2, 0, -3).is_some());
        assert!(Swizzle::try_new(3, 30, 3).is_none());
    }

    #[test]
    #[should_panic(expected = "invalid CuTe swizzle")]
    fn unchecked_construction_fails_loudly() {
        let _ = Swizzle::new(4, 3, 3);
    }

    #[test]
    fn identity_is_identity() {
        for x in 0..256 {
            assert_eq!(Swizzle::IDENTITY.apply(x), x);
        }
    }

    #[test]
    fn involution_on_disjoint_fields() {
        // Because the source and target do not overlap, applying the same XOR
        // permutation twice restores every offset.
        let s = Swizzle::new(3, 3, 3);
        assert!(s.is_involution());
        for x in 0..1024 {
            assert_eq!(s.apply(s.apply(x)), x, "double apply must restore {x}");
        }

        // Every input in this complete bit window maps to one distinct output:
        // the swizzle reorders offsets but never aliases two of them.
        let mut seen: alloc::vec::Vec<i64> = (0..512).map(|x| s.apply(x)).collect();
        seen.sort_unstable();
        assert_eq!(seen, (0..512).collect::<alloc::vec::Vec<_>>());
    }

    #[test]
    fn base_bits_pass_through() {
        let s = Swizzle::new(3, 4, 3);
        for x in 0..2048 {
            assert_eq!(
                s.apply(x) & 0b1111,
                x & 0b1111,
                "low base bits must not change"
            );
        }
    }

    #[test]
    fn masks_and_alignment_match_the_bit_fields() {
        let s = Swizzle::new(3, 4, 3);
        assert_eq!(s.y_mask(), 0b111 << 7);
        assert_eq!(s.z_mask(), 0b111 << 4);
        assert_eq!(s.max_alignment(), 16);
    }

    #[test]
    fn cuda_device_conversion_is_explicit_and_positive_only() {
        assert_eq!(
            Swizzle::new(3, 3, 3).cuda_device_absolute_source_bit(),
            Some(6)
        );
        assert_eq!(
            Swizzle::new(2, 0, 2).cuda_device_absolute_source_bit(),
            Some(2)
        );
        assert_eq!(
            Swizzle::new(2, 0, -3).cuda_device_absolute_source_bit(),
            None
        );
    }
}
