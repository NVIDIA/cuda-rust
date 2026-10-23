/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compatibility for the published 0.1.0 printer's missing argument locations.
//!
//! Upstream owns operation syntax, SSA names, types, and location syntax. Its
//! private block printer has no location hook, so annotate only complete block
//! headers derived from the same graph after printing. Require each affected
//! header exactly once; upstream format changes must produce a diagnostic.
//! Checked dialect tokens contain no newlines, and upstream escapes newlines
//! in string literals, so a complete header line cannot be attribute content.
//! Remove this bridge when upstream exposes argument-location printing.

use std::{collections::HashMap, fmt};

use pliron::{
    basic_block::BasicBlock,
    common_traits::Named,
    context::{Context, Ptr},
    location::Location,
    printable::State,
    r#type::Typed,
    value::Value,
};
use pliron_mlir_interop::{
    MlirPrinter, MlirPrinterT,
    printers::{print_op_location, print_type},
};

use super::tokens::printer_error;
use crate::TranslationError;

/// Render one graph-derived header, optionally including known argument locations.
struct BlockHeader<'a> {
    /// Target block whose unique names and actual types must match upstream output.
    block: Ptr<BasicBlock>,
    /// Known locations; absent entries retain upstream's implicit unknown location.
    locations: Option<&'a HashMap<Value, &'a Location>>,
}

impl MlirPrinterT for BlockHeader<'_> {
    fn mlir_print(
        &self,
        ctx: &Context,
        state: &State,
        f: &mut fmt::Formatter<'_>,
    ) -> pliron::result::Result<()> {
        let block = self.block.deref(ctx);
        write!(f, "^{}(", block.unique_name(ctx))?;
        for (index, argument) in block.arguments().enumerate() {
            if index > 0 {
                write!(f, ", ")?;
            }
            write!(f, "%{}: ", argument.unique_name(ctx))?;
            print_type(ctx, argument.get_type(ctx), state, f)?;
            if let Some(location) = self
                .locations
                .and_then(|locations| locations.get(&argument))
            {
                print_op_location(ctx, location, state, f)?;
            }
        }
        write!(f, "):")?;
        Ok(())
    }
}

/// Restore known argument locations, rejecting any unexpected upstream header format.
pub(super) fn restore(
    ctx: &Context,
    text: String,
    blocks: impl IntoIterator<Item = Ptr<BasicBlock>>,
    argument_locations: &[(Value, Location)],
) -> Result<String, TranslationError> {
    let locations: HashMap<_, _> = argument_locations
        .iter()
        .filter(|(_, location)| !location.is_unknown())
        .map(|(value, location)| (*value, location))
        .collect();
    let mut headers = HashMap::new();
    for block in blocks {
        if !block
            .deref(ctx)
            .arguments()
            .any(|argument| locations.contains_key(&argument))
        {
            continue;
        }
        let mut header = BlockHeader {
            block,
            locations: None,
        };
        let original = String::try_from(MlirPrinter::new(ctx, &header))
            .map_err(|error| printer_error(ctx, error))?;
        header.locations = Some(&locations);
        let annotated = String::try_from(MlirPrinter::new(ctx, &header))
            .map_err(|error| printer_error(ctx, error))?;
        headers.insert(original, (annotated, false));
    }
    replace_headers(text, headers)
}

/// Replace complete header lines exactly once, preserving all other bytes.
fn replace_headers(
    text: String,
    mut headers: HashMap<String, (String, bool)>,
) -> Result<String, TranslationError> {
    if headers.is_empty() {
        return Ok(text);
    }
    let mut output = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        let body = line.strip_suffix('\n').unwrap_or(line);
        let header = body.trim_start_matches(' ');
        if let Some((annotated, seen)) = headers.get_mut(header) {
            if *seen {
                return Err(TranslationError::InvalidTarget(
                    "upstream MLIR block header occurs more than once while restoring argument locations"
                        .into(),
                ));
            }
            *seen = true;
            output.push_str(&body[..body.len() - header.len()]);
            output.push_str(annotated);
            if line.ends_with('\n') {
                output.push('\n');
            }
        } else {
            output.push_str(line);
        }
    }
    if headers.values().any(|(_, seen)| !seen) {
        return Err(TranslationError::InvalidTarget(
            "upstream MLIR block header changed while restoring argument locations".into(),
        ));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::replace_headers;

    #[test]
    fn unexpected_header_format_is_rejected_instead_of_dropping_locations() {
        let headers = [(
            "^block1v1(%v0: i32):".into(),
            ("^block1v1(%v0: i32 loc(\"argument\")):".into(), false),
        )]
        .into();
        assert!(
            replace_headers("  ^block1v1(%v0 : i32):\n".into(), headers)
                .unwrap_err()
                .to_string()
                .contains("block header changed")
        );
    }

    #[test]
    fn duplicate_headers_are_rejected_instead_of_annotating_both() {
        let headers = [(
            "^block1v1(%v0: i32):".into(),
            ("^block1v1(%v0: i32 loc(\"argument\")):".into(), false),
        )]
        .into();
        assert!(
            replace_headers(
                "  ^block1v1(%v0: i32):\n    ^block1v1(%v0: i32):\n".into(),
                headers,
            )
            .unwrap_err()
            .to_string()
            .contains("block header occurs more than once")
        );
    }
}
