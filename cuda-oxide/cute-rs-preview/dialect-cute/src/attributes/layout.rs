/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Structural layouts and swizzles in the CuTe attribute syntax.

use core::fmt;

use combine::Parser;
use cute_layout::{ComposedLayout, Layout, OffsetUnit, ParseLayoutError, Swizzle};
use pliron::{
    impl_printable_for_display,
    irfmt::parsers::quoted_string_parser,
    parsable::{Parsable, ParseResult, StateStream},
};
use pliron_derive::pliron_attr;

/// A complete swizzled layout, stored as one typed attribute.
///
/// The quoted payload is intentionally simple and round-trippable. Its
/// fields are CuTe's inner swizzle, offset, unit, and outer layout, in that
/// order:
///
/// ```text
/// "3,4,3;0;elements;(8,32):(32,1)"
///   B M S  off unit      outer layout
/// ```
#[pliron_attr(name = "cute.composed_layout", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct CuteComposedLayoutAttr(
    #[doc = "Structural inner swizzle, offset, and outer layout."] pub ComposedLayout,
);

impl fmt::Display for CuteComposedLayoutAttr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let swizzle = self.0.inner();
        write!(
            f,
            "\"{},{},{};{};{};{}\"",
            swizzle.num_bits,
            swizzle.num_base,
            swizzle.num_shift,
            self.0.offset(),
            self.0.unit(),
            self.0.outer()
        )
    }
}

impl_printable_for_display!(CuteComposedLayoutAttr);

/// Parse the structural inner swizzle, offset, unit, and outer layout from the printed attribute syntax.
pub(super) fn parse_composed_layout(
    text: String,
) -> Result<CuteComposedLayoutAttr, ParseLayoutError> {
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
    let outer = fields.next().ok_or(ParseLayoutError)?.parse::<Layout>()?;
    let swizzle_fields = swizzle.split(',').map(str::trim).collect::<Vec<_>>();
    let [bits, base, shift] = swizzle_fields.as_slice() else {
        return Err(ParseLayoutError);
    };
    let bits = bits.parse::<u32>().map_err(|_| ParseLayoutError)?;
    let base = base.parse::<u32>().map_err(|_| ParseLayoutError)?;
    let shift = shift.parse::<i32>().map_err(|_| ParseLayoutError)?;
    let inner = Swizzle::try_new(bits, base, shift).ok_or(ParseLayoutError)?;
    ComposedLayout::new(inner, offset, outer, unit)
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

/// A CuTe layout (`shape:stride`) as a first-class typed attribute.
#[pliron_attr(name = "cute.layout", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct CuteLayoutAttr(#[doc = "Structural shape and stride mapping."] pub Layout);

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
