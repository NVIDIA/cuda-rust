/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Text formatting and parsing for integer tuples and `shape:stride` layouts.

use alloc::vec::Vec;
use core::fmt;
use core::str::FromStr;

use crate::{IntTuple, Layout};

// Text form used in diagnostics and in the cutegen test oracle:
//
//   shape:stride
//   (8,32):(32,1)
//   ((2,4),32):((128,32),1)  // nested mode on the left
//
// Parentheses preserve the tuple tree; they are not merely decoration.

impl fmt::Display for IntTuple {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntTuple::Leaf(v) => write!(f, "{v}"),
            IntTuple::Tuple(ts) => {
                write!(f, "(")?;
                for (i, t) in ts.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{t}")?;
                }
                write!(f, ")")
            }
        }
    }
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.shape, self.stride)
    }
}

/// Indicates that text was not a complete, structurally valid layout or tuple.
///
/// Examples include a missing `:`, an unclosed tuple, or shape and stride
/// trees that do not match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseLayoutError;

impl fmt::Display for ParseLayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid CuTe layout/tuple syntax")
    }
}

impl core::error::Error for ParseLayoutError {}

/// Parse one integer or parenthesized tuple from the remaining layout text.
fn parse_int_tuple(s: &str) -> Result<(IntTuple, &str), ParseLayoutError> {
    let s = s.trim_start();
    if let Some(mut rest) = s.strip_prefix('(') {
        let mut items = Vec::new();
        loop {
            let (item, r) = parse_int_tuple(rest)?;
            items.push(item);
            let r = r.trim_start();
            if let Some(r) = r.strip_prefix(',') {
                rest = r;
            } else if let Some(r) = r.strip_prefix(')') {
                return Ok((IntTuple::Tuple(items), r));
            } else {
                return Err(ParseLayoutError);
            }
        }
    } else {
        let end = s
            .char_indices()
            .find(|(_, c)| !(c.is_ascii_digit() || *c == '-'))
            .map(|(i, _)| i)
            .unwrap_or(s.len());
        let value: i64 = s[..end].parse().map_err(|_| ParseLayoutError)?;
        Ok((IntTuple::Leaf(value), &s[end..]))
    }
}

impl FromStr for IntTuple {
    type Err = ParseLayoutError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (t, rest) = parse_int_tuple(s)?;
        if rest.trim().is_empty() {
            Ok(t)
        } else {
            Err(ParseLayoutError)
        }
    }
}

impl FromStr for Layout {
    type Err = ParseLayoutError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (shape, rest) = parse_int_tuple(s)?;
        let rest = rest.trim_start();
        let rest = rest.strip_prefix(':').ok_or(ParseLayoutError)?;
        let (stride, rest) = parse_int_tuple(rest)?;
        if !rest.trim().is_empty() {
            return Err(ParseLayoutError);
        }
        if !shape.congruent(&stride) {
            return Err(ParseLayoutError);
        }
        Ok(Layout { shape, stride })
    }
}
