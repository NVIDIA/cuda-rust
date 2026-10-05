/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Source-location helpers for MIR translation.
//!
//! rustc gives us source spans while we still know which MIR statement or
//! terminator produced an op. Store those spans as structured pliron locations
//! here, before later lowering stages only see generic operations.

use std::path::PathBuf;

use combine::stream::position::SourcePosition;
use pliron::context::Context;
use pliron::location::{Location, Source};
use rustc_public::ty::Span;

/// Convert a rustc source span into a pliron source position.
///
/// A span with no real file or no positive line number cannot produce useful
/// line-table debug info, so it becomes `Unknown` and the LLVM exporter skips
/// `!dbg` for that operation.
pub(crate) fn span_to_location(ctx: &mut Context, span: Span) -> Location {
    let file = span.get_filename();
    let lines = span.get_lines();

    if file.is_empty() || lines.start_line == 0 || lines.start_col == 0 {
        return Location::Unknown;
    }

    Location::SrcPos {
        src: Source::new_from_file(ctx, PathBuf::from(file)),
        pos: SourcePosition {
            line: lines.start_line as i32,
            column: lines.start_col as i32,
        },
    }
}
