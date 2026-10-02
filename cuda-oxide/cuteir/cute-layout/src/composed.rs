/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Keep a layout and its swizzle together as one offset calculation.
//!
//! A [`Layout`] answers a basic question: “where does this logical coordinate
//! live relative to the start of a tile?” For example, a row-major 2-by-4
//! layout maps `(row, column)` like this:
//!
//! ```text
//! logical coordinates              plain element offsets
//! (0,0) (0,1) (0,2) (0,3)             0  1  2  3
//! (1,0) (1,1) (1,2) (1,3)             4  5  6  7
//! ```
//!
//! A [`Swizzle`] then permutes offset bits, usually to spread shared-memory
//! requests across hardware banks. [`ComposedLayout`] stores both operations
//! so a caller cannot accidentally use the layout but forget its swizzle:
//!
//! ```text
//! coordinate
//!     │
//!     ▼
//! inner layout ──> plain offset ──> + static offset ──> outer swizzle
//!                                                               │
//!                                                               ▼
//!                                                  physical tile offset
//!
//! call(c) = outer(offset + inner(c))
//! ```
//!
//! The names “inner” and “outer” describe evaluation order: the inner layout
//! runs first; the outer swizzle runs last. The stored `offset` is a fixed
//! displacement between them.
//!
//! ## Concrete example
//!
//! Start with row and column [modes](crate#modes), then swizzle the plain
//! row-major offsets:
//!
//! ```text
//! inner layout: (2,4):(4,1)
//! row mode:      extent 2, stride 4
//! column mode:   extent 4, stride 1
//! fixed offset:  0
//! outer swizzle: S<1,0,2>  (offset bit 2 toggles bit 0)
//!
//! coordinates             plain offsets       final offsets
//! (0,0) (0,1) (0,2) (0,3)   0  1  2  3    ->   0  1  2  3
//! (1,0) (1,1) (1,2) (1,3)   4  5  6  7    ->   5  4  7  6
//! ```
//!
//! Every result is still a **tile-relative offset**, not an absolute pointer.
//! The allocation's base pointer is added only after this calculation:
//!
//! ```text
//! absolute address = allocation base pointer + physical tile offset
//! ```
//!
//! Keeping the pointer outside the XOR makes the same tile mapping work no
//! matter where the allocation happens to begin.
//!
//! ## Elements and bytes are different units
//!
//! An *element offset* counts typed values: for `f32`, element offset 3 means
//! the fourth `f32`. A *byte offset* counts individual bytes: that same `f32`
//! begins at byte offset `3 × 4 = 12`.
//!
//! This distinction matters because a swizzle operates on the binary bits of
//! the offset. Bit 0 means one element in an element-unit layout, but one byte
//! in a byte-unit layout. [`OffsetUnit`] records which interpretation applies.
//! Use [`ComposedLayout::to_byte_offsets`] before byte-addressed validation or
//! code generation.

use core::fmt;

use crate::algebra::{AlgebraError, max_alignment};
use crate::{IntTuple, Layout, Swizzle};

/// What one step in every stored and returned offset represents.
///
/// The inner layout's strides, the fixed composed offset, and the number fed
/// to the swizzle must all use this same unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OffsetUnit {
    /// One step means one typed value, such as one `f32`.
    ///
    /// The element type and its byte size are supplied later by the operation
    /// that uses the layout.
    Elements,
    /// One step means one byte, so the result can be added to a byte pointer.
    Bytes,
}

impl fmt::Display for OffsetUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OffsetUnit::Elements => write!(f, "elements"),
            OffsetUnit::Bytes => write!(f, "bytes"),
        }
    }
}

/// Why a composed layout could not be built or converted to byte offsets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComposedLayoutError {
    /// The swizzle's source and target fields overlap or fall outside the
    /// supported low 32-bit swizzle window.
    InvalidSwizzle,
    /// Element-to-byte conversion needs a positive power-of-two byte size.
    /// The payload is the rejected size.
    InvalidElementBytes(i64),
    /// Scaling the offset or a stride exceeded the `i64` offset model.
    ArithmeticOverflow,
}

impl fmt::Display for ComposedLayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComposedLayoutError::InvalidSwizzle => {
                write!(f, "composed layout has an invalid CuTe swizzle")
            }
            ComposedLayoutError::InvalidElementBytes(bytes) => write!(
                f,
                "element size must be a positive power of two, got {bytes} bytes"
            ),
            ComposedLayoutError::ArithmeticOverflow => {
                write!(f, "composed-layout unit conversion overflowed")
            }
        }
    }
}

impl core::error::Error for ComposedLayoutError {}

/// A complete, position-independent mapping from coordinates to swizzled
/// tile-relative offsets.
///
/// It stores all three parts of `outer(offset + inner(coordinate))`, plus the
/// unit in which those offsets are measured. It deliberately does not store
/// the allocation's base pointer.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ComposedLayout {
    outer: Swizzle,
    offset: i64,
    inner: Layout,
    unit: OffsetUnit,
}

impl ComposedLayout {
    /// Build the mapping `outer(offset + inner(coordinate))`.
    ///
    /// `offset`, every stride in `inner`, and the input expected by `outer`
    /// must already use `unit`. Construction rejects a swizzle whose bit
    /// fields overlap or fall outside the supported low 32-bit swizzle window.
    /// Higher offset bits pass through unchanged. See the
    /// module's [concrete example](crate::composed#concrete-example) for all
    /// three steps with numbers.
    pub fn new(
        outer: Swizzle,
        offset: i64,
        inner: Layout,
        unit: OffsetUnit,
    ) -> Result<Self, ComposedLayoutError> {
        if !outer.is_valid() {
            return Err(ComposedLayoutError::InvalidSwizzle);
        }
        Ok(ComposedLayout {
            outer,
            offset,
            inner,
            unit,
        })
    }

    /// Turn a plain layout into a composition that produces the same offsets.
    ///
    /// The identity swizzle changes no bits and the fixed offset is zero:
    ///
    /// ```text
    /// inner layout (2,3):(3,1), coordinate (1,2)
    /// inner offset 5 -> identity swizzle -> final offset 5
    /// ```
    pub fn from_layout(inner: Layout, unit: OffsetUnit) -> Self {
        ComposedLayout {
            outer: Swizzle::IDENTITY,
            offset: 0,
            inner,
            unit,
        }
    }

    /// Return the swizzle applied last.
    ///
    /// “Outer” describes composition order; it does not mean an outer tile or
    /// an absolute address.
    pub const fn outer(&self) -> Swizzle {
        self.outer
    }

    /// Return the fixed displacement added after the inner layout and before
    /// the swizzle, measured in [`Self::unit`].
    pub const fn offset(&self) -> i64 {
        self.offset
    }

    /// Return the plain coordinate-to-offset layout evaluated first.
    pub const fn inner(&self) -> &Layout {
        &self.inner
    }

    /// Return whether this composition's offsets count elements or bytes.
    pub const fn unit(&self) -> OffsetUnit {
        self.unit
    }

    /// Return the number of logical coordinates accepted by the inner layout.
    ///
    /// Swizzling only permutes locations, so it does not change this count.
    pub fn size(&self) -> i64 {
        self.inner.size()
    }

    /// Map one logical coordinate to a physical tile-relative offset.
    ///
    /// The returned number uses [`Self::unit`]. This unchecked form assumes
    /// that `coord` has the right structure and lies inside the inner layout;
    /// compiler input should use [`Self::checked_call`] instead. The module's
    /// [concrete example](crate::composed#concrete-example) shows the complete
    /// inner-layout, fixed-offset, and swizzle calculation.
    pub fn call(&self, coord: &IntTuple) -> i64 {
        self.call_with_inner_delta(coord, 0)
    }

    /// Map an in-range coordinate without panicking or overflowing.
    ///
    /// Returns `None` when the coordinate has the wrong structure, lies
    /// outside the layout, or either addition exceeds `i64`. Compiler passes
    /// use this for layouts decoded from the compiler's intermediate
    /// representation (IR), which must be treated as untrusted input.
    pub fn checked_call(&self, coord: &IntTuple) -> Option<i64> {
        self.checked_call_with_inner_delta(coord, 0)
    }

    /// Map a position shortly after the item selected by `coord`.
    ///
    /// `delta` is added to the inner layout's result before swizzling and uses
    /// the composition's declared unit. For a byte-unit layout, `delta = 3`
    /// means “three bytes after this item's first byte.”
    ///
    /// For a one-thread [copy atom](crate#atoms) such as `CpAsync<16>`,
    /// validation calls this method for deltas 0 through 15 to check that the
    /// swizzle keeps those 16 bytes together and in order.
    pub fn call_with_inner_delta(&self, coord: &IntTuple, delta: i64) -> i64 {
        self.outer
            .apply(self.offset + self.inner.call(coord) + delta)
    }

    /// Checked form of [`Self::call_with_inner_delta`].
    ///
    /// Returns `None` for an invalid coordinate or arithmetic overflow.
    pub fn checked_call_with_inner_delta(&self, coord: &IntTuple, delta: i64) -> Option<i64> {
        let input = self
            .offset
            .checked_add(self.inner.checked_call(coord)?)?
            .checked_add(delta)?;
        Some(self.outer.apply(input))
    }

    /// Convert an element-counting composition into an equivalent
    /// byte-counting composition.
    ///
    /// For a power-of-two element size `2^k`, converting an element offset to
    /// bytes shifts its binary value left by `k` places. Both swizzle fields
    /// must move left by the same amount, so `M` increases by `k` while `B`
    /// and `S` remain unchanged:
    ///
    /// ```text
    /// four-byte elements: k = 2
    ///
    /// element S<2,2,3>: [source 6..5][gap 4][target 3..2][unchanged 1..0]
    /// byte    S<2,4,3>: [source 8..7][gap 6][target 5..4][unchanged 3..0]
    ///
    /// S<2,2,3> in elements  ──>  S<2,4,3> in bytes
    /// ```
    ///
    /// The layout strides change units too:
    ///
    /// ```text
    /// f32 layout:       (2,3):(3,1) elements -> (2,3):(12,4) bytes
    /// coordinate (1,2): 5 elements          -> 20 bytes
    /// ```
    ///
    /// The fixed offset and every inner stride are multiplied by
    /// `element_bytes` as well. The resulting physical byte offset is exactly
    /// the old physical element offset multiplied by `element_bytes`.
    ///
    /// A byte-unit composition is returned unchanged. Zero, negative, and
    /// non-power-of-two element sizes are rejected because a simple bit-field
    /// shift cannot represent them. Overflow and a shifted swizzle whose
    /// fields leave the supported low 32-bit swizzle window are also reported.
    pub fn to_byte_offsets(&self, element_bytes: i64) -> Result<Self, ComposedLayoutError> {
        if self.unit == OffsetUnit::Bytes {
            return Ok(self.clone());
        }
        if element_bytes <= 0 || !(element_bytes as u64).is_power_of_two() {
            return Err(ComposedLayoutError::InvalidElementBytes(element_bytes));
        }

        let added_bits = element_bytes.trailing_zeros();
        let new_base = self
            .outer
            .base
            .checked_add(added_bits)
            .ok_or(ComposedLayoutError::ArithmeticOverflow)?;
        let outer = Swizzle::try_new(self.outer.bits, new_base, self.outer.shift)
            .ok_or(ComposedLayoutError::InvalidSwizzle)?;
        let offset = self
            .offset
            .checked_mul(element_bytes)
            .ok_or(ComposedLayoutError::ArithmeticOverflow)?;
        let stride = checked_scale_tuple(&self.inner.stride, element_bytes)
            .ok_or(ComposedLayoutError::ArithmeticOverflow)?;

        Ok(ComposedLayout {
            outer,
            offset,
            inner: Layout::new(self.inner.shape.clone(), stride),
            unit: OffsetUnit::Bytes,
        })
    }

    /// Return a conservative access-alignment factor supported by every part
    /// of this composition, measured in [`Self::unit`].
    ///
    /// Alignment means an access may safely begin only at a multiple of some
    /// value. A composition cannot promise more alignment than its weakest
    /// part, so the result is the greatest common divisor of:
    ///
    /// ```text
    /// inner layout's alignment
    /// fixed offset
    /// swizzle's chunk-alignment factor
    ///                │
    ///                ▼
    /// gcd(inner alignment, fixed offset, swizzle alignment)
    /// ```
    ///
    /// Example: an inner layout aligned to 16 bytes plus an 8-byte fixed
    /// offset can promise only 8-byte alignment, even if the swizzle preserves
    /// 16-byte chunks.
    ///
    /// A swizzle with zero XOR bits changes no offset at all, so it adds no
    /// restriction regardless of its placeholder `M` field; only a real
    /// swizzle caps the result at its `2^M` preserved-chunk width. This is a
    /// static layout property; the real allocation pointer must still be
    /// suitably aligned.
    pub fn max_alignment(&self) -> Result<i64, AlgebraError> {
        let inner_alignment = max_alignment(&self.inner)?;
        // gcd(x, 0) = x, so a zero offset drops out on its own.
        let with_offset = gcd(inner_alignment, unsigned_to_i64(self.offset.unsigned_abs()));
        if self.outer.bits == 0 {
            return Ok(with_offset.max(1));
        }
        Ok(gcd(with_offset, self.outer.max_alignment()).max(1))
    }
}

impl fmt::Display for ComposedLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} o {} o {} [{}]",
            self.outer, self.offset, self.inner, self.unit
        )
    }
}

fn checked_scale_tuple(tuple: &IntTuple, scale: i64) -> Option<IntTuple> {
    match tuple {
        IntTuple::Leaf(value) => value.checked_mul(scale).map(IntTuple::Leaf),
        IntTuple::Tuple(items) => items
            .iter()
            .map(|item| checked_scale_tuple(item, scale))
            .collect::<Option<_>>()
            .map(IntTuple::Tuple),
    }
}

fn gcd(a: i64, b: i64) -> i64 {
    let mut a = a.unsigned_abs();
    let mut b = b.unsigned_abs();
    while b != 0 {
        (a, b) = (b, a % b);
    }
    unsigned_to_i64(a)
}

fn unsigned_to_i64(value: u64) -> i64 {
    // Valid layout values normally fit after taking their absolute value.
    // `i64::MIN` is the sole exception; returning 1 makes no unsafe alignment
    // promise when that malformed edge case reaches this helper.
    i64::try_from(value).unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    fn l(text: &str) -> Layout {
        text.parse().unwrap()
    }

    #[test]
    fn call_uses_cutegen_composition_order() {
        let layout =
            ComposedLayout::new(Swizzle::new(2, 0, 3), 8, l("4:1"), OffsetUnit::Bytes).unwrap();

        for coord in 0..4 {
            let coord = IntTuple::Leaf(coord);
            assert_eq!(
                layout.call(&coord),
                layout.outer.apply(8 + layout.inner.call(&coord))
            );
        }
        assert_eq!(layout.to_string(), "S<2,0,3> o 8 o 4:1 [bytes]");
    }

    #[test]
    fn byte_conversion_preserves_the_physical_mapping() {
        let elements = ComposedLayout::new(
            Swizzle::new(2, 2, 3),
            8,
            l("(4,4):(1,4)"),
            OffsetUnit::Elements,
        )
        .unwrap();
        let bytes = elements.to_byte_offsets(4).unwrap();

        assert_eq!(bytes.unit(), OffsetUnit::Bytes);
        assert_eq!(bytes.outer(), Swizzle::new(2, 4, 3));
        assert_eq!(bytes.offset(), 32);
        assert_eq!(bytes.inner().to_string(), "(4,4):(4,16)");
        for coord in 0..elements.size() {
            let coord = IntTuple::Leaf(coord);
            assert_eq!(bytes.call(&coord), elements.call(&coord) * 4);
        }
    }

    #[test]
    fn byte_conversion_rejects_unrepresentable_sizes() {
        let layout = ComposedLayout::from_layout(l("4:1"), OffsetUnit::Elements);
        assert_eq!(
            layout.to_byte_offsets(3),
            Err(ComposedLayoutError::InvalidElementBytes(3))
        );

        let high_bits =
            ComposedLayout::new(Swizzle::new(1, 30, 1), 0, l("4:1"), OffsetUnit::Elements).unwrap();
        assert_eq!(
            high_bits.to_byte_offsets(2),
            Err(ComposedLayoutError::InvalidSwizzle)
        );
    }

    #[test]
    fn construction_revalidates_deserialized_swizzles() {
        // Compiler code can decode a Swizzle directly from its intermediate
        // representation, bypassing Swizzle::new. ComposedLayout is therefore
        // a second input boundary and rejects overlapping bit fields.
        let invalid = Swizzle {
            bits: 4,
            base: 3,
            shift: 3,
        };
        assert_eq!(
            ComposedLayout::new(invalid, 0, l("4:1"), OffsetUnit::Bytes),
            Err(ComposedLayoutError::InvalidSwizzle)
        );
    }

    #[test]
    fn checked_call_rejects_offset_overflow() {
        let layout =
            ComposedLayout::new(Swizzle::IDENTITY, i64::MAX, l("2:1"), OffsetUnit::Bytes).unwrap();
        assert_eq!(layout.checked_call(&IntTuple::Leaf(1)), None);
    }

    #[test]
    fn max_alignment_combines_every_part() {
        let layout = ComposedLayout::new(
            Swizzle::new(3, 4, 4),
            0,
            l("(32,1):(1,32)"),
            OffsetUnit::Bytes,
        )
        .unwrap();
        assert_eq!(layout.max_alignment().unwrap(), 16);

        let offset_limited = ComposedLayout::new(
            Swizzle::new(3, 4, 4),
            8,
            l("(32,1):(1,32)"),
            OffsetUnit::Bytes,
        )
        .unwrap();
        assert_eq!(offset_limited.max_alignment().unwrap(), 8);

        let inner_limited = ComposedLayout::new(
            Swizzle::new(3, 4, 4),
            0,
            l("(8,16):(1,4)"),
            OffsetUnit::Bytes,
        )
        .unwrap();
        assert_eq!(inner_limited.max_alignment().unwrap(), 4);
    }

    #[test]
    fn zero_bit_swizzle_adds_no_alignment_cap() {
        // Swizzle::IDENTITY is S<0,4,3>: its M = 4 is a placeholder, and with
        // zero XOR bits it changes no offset. The offset must still count.
        let offset_only =
            ComposedLayout::new(Swizzle::IDENTITY, 32, l("128:1"), OffsetUnit::Bytes).unwrap();
        assert_eq!(offset_only.max_alignment().unwrap(), 32);

        let unrestricted =
            ComposedLayout::new(Swizzle::IDENTITY, 0, l("128:1"), OffsetUnit::Bytes).unwrap();
        assert_eq!(unrestricted.max_alignment().unwrap(), 128);
    }
}
