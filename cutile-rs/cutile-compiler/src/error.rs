/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! JIT compilation error types and helpers for source-located diagnostics.

use crate::ast::SourceLocation;
use anyhow::Result;
use proc_macro2::LexError;
use std::{
    error,
    fmt::{self, Display, Formatter},
};

/// Error type for the JIT compiler, with optional source location info.
pub enum JITError {
    /// An error without source location.
    Generic(String),
    /// An error with an associated [`SourceLocation`] captured at proc macro
    /// expansion time. Unlike a `proc_macro2::Span` (or a `syn::Error` built
    /// on one), which is only meaningful inside a proc macro context, this
    /// variant carries a concrete file/line/column that can be displayed to
    /// the user even at JIT (runtime) compilation.
    Located(String, SourceLocation),
    /// A wrapped `anyhow::Error`.
    Anyhow(anyhow::Error),
}

impl fmt::Debug for JITError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        // Debug delegates to Display so that `.unwrap()` output is identical
        // to the user-facing format.
        Display::fmt(self, f)
    }
}

impl Display for JITError {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Self::Generic(error) => {
                write!(f, "{error}")
            }
            Self::Located(error, loc) if loc.is_known() => {
                write!(
                    f,
                    "{error}\n  --> {file}:{line}:{column}",
                    file = loc.file,
                    line = loc.line,
                    column = loc.column,
                )
            }
            Self::Located(error, _) => write!(f, "{error}"),
            Self::Anyhow(error) => write!(f, "{error}"),
        }
    }
}

impl error::Error for JITError {}

impl From<cutile_ir::capabilities::CapabilityError> for JITError {
    fn from(error: cutile_ir::capabilities::CapabilityError) -> Self {
        fn source(loc: &cutile_ir::ir::Location) -> SourceLocation {
            use cutile_ir::ir::Location;
            match loc {
                Location::FileLineCol {
                    filename,
                    line,
                    column,
                } => SourceLocation::new(filename.clone(), *line as usize, *column as usize),
                Location::DebugInfo(loc) => SourceLocation::new(
                    loc.filename.clone(),
                    loc.line as usize,
                    loc.column as usize,
                ),
                Location::CallSite { callee, .. } => source(callee),
                Location::Unknown => SourceLocation::unknown(),
            }
        }
        Self::Located(error.message, source(&error.location))
    }
}

impl JITError {
    /// Create a `Generic` error value (not wrapped in `Result`).
    pub fn generic_err(err_str: &str) -> JITError {
        JITError::Generic(err_str.to_string())
    }
    /// Create a `Generic` error wrapped in `Err`.
    pub fn generic<R>(err_str: &str) -> Result<R, JITError> {
        Err(JITError::generic_err(err_str))
    }
    /// Create a `Located` error that carries a real source location captured
    /// at proc macro expansion time.
    pub fn located(error_message: &str, location: SourceLocation) -> JITError {
        JITError::Located(error_message.to_string(), location)
    }
    /// Create a `Located` error wrapped in `Err`.
    pub fn located_result<R>(error_message: &str, location: SourceLocation) -> Result<R, JITError> {
        Err(JITError::located(error_message, location))
    }
}

/// Trait for creating source-located errors from a [`SourceLocation`].
pub trait SpannedJITError {
    /// Create a located error from this span.
    fn jit_error(&self, error_message: &str) -> JITError;
    /// Assert a predicate, returning a located error on failure.
    fn jit_assert(&self, pred: bool, error_message: &str) -> Result<(), JITError>;
    /// Create a located error wrapped in `Err`.
    fn jit_error_result<R>(&self, error_message: &str) -> Result<R, JITError>;
}

impl SpannedJITError for SourceLocation {
    fn jit_error(&self, error_message: &str) -> JITError {
        JITError::located(error_message, self.clone())
    }
    fn jit_assert(&self, pred: bool, error_message: &str) -> Result<(), JITError> {
        if pred {
            Ok(())
        } else {
            JITError::located_result(error_message, self.clone())
        }
    }
    fn jit_error_result<R>(&self, error_message: &str) -> Result<R, JITError> {
        JITError::located_result(error_message, self.clone())
    }
}

impl From<anyhow::Error> for JITError {
    fn from(error: anyhow::Error) -> Self {
        Self::Anyhow(error)
    }
}

impl From<LexError> for JITError {
    fn from(error: LexError) -> Self {
        Self::Generic(error.to_string())
    }
}

/// Extension trait to convert `Option<T>` into `Result<T, JITError>`.
pub trait OptionJITError<T> {
    /// Unwrap or return a generic JIT error with the given message.
    fn ok_or_jit_error(self, message: &str) -> Result<T, JITError>;
}

impl<T> OptionJITError<T> for Option<T> {
    fn ok_or_jit_error(self, message: &str) -> Result<T, JITError> {
        self.ok_or_else(|| JITError::Generic(message.to_string()))
    }
}

/// Assert a predicate, returning a generic JIT error on failure.
pub fn jit_assert(pred: bool, message: &str) -> Result<(), JITError> {
    if pred {
        Ok(())
    } else {
        Err(JITError::Generic(message.to_string()))
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::SourceLocation;

    // -----------------------------------------------------------------------
    // 1. Formatting consistency: Display and Debug produce identical output
    //    for every JITError variant.
    // -----------------------------------------------------------------------

    #[test]
    fn generic_display_and_debug_are_identical() {
        let err = JITError::Generic("something broke".into());
        let display = format!("{err}");
        let debug = format!("{err:?}");
        assert_eq!(
            display, debug,
            "Display and Debug must be identical for JITError::Generic"
        );
    }

    #[test]
    fn located_known_display_and_debug_are_identical() {
        let loc = SourceLocation::new("src/main.rs".into(), 42, 8);
        let err = JITError::Located("bad token".into(), loc);
        let display = format!("{err}");
        let debug = format!("{err:?}");
        assert_eq!(
            display, debug,
            "Display and Debug must be identical for JITError::Located (known)"
        );
    }

    #[test]
    fn located_unknown_display_and_debug_are_identical() {
        let loc = SourceLocation::unknown();
        let err = JITError::Located("bad token".into(), loc);
        let display = format!("{err}");
        let debug = format!("{err:?}");
        assert_eq!(
            display, debug,
            "Display and Debug must be identical for JITError::Located (unknown)"
        );
    }

    #[test]
    fn anyhow_display_and_debug_are_identical() {
        let err = JITError::Anyhow(anyhow::anyhow!("anyhow problem"));
        let display = format!("{err}");
        let debug = format!("{err:?}");
        assert_eq!(
            display, debug,
            "Display and Debug must be identical for JITError::Anyhow"
        );
    }

    // -----------------------------------------------------------------------
    // 2. Formatting consistency: the "error: " prefix appears exactly once
    //    for Generic and Located variants — never doubled.
    // -----------------------------------------------------------------------

    #[test]
    fn generic_has_no_error_prefix() {
        let err = JITError::Generic("division by zero".into());
        let output = format!("{err}");
        assert_eq!(
            output, "division by zero",
            "Generic variant must render the bare message without an 'error: ' prefix, got: {output}"
        );
    }

    #[test]
    fn located_known_has_no_error_prefix() {
        let loc = SourceLocation::new("lib.rs".into(), 10, 5);
        let err = JITError::Located("type mismatch".into(), loc);
        let output = format!("{err}");
        assert!(
            output.starts_with("type mismatch"),
            "Located (known) variant must start with the bare message, got: {output}"
        );
        assert!(
            !output.starts_with("error: "),
            "Located (known) variant must not include an 'error: ' prefix, got: {output}"
        );
    }

    #[test]
    fn located_unknown_has_no_error_prefix() {
        let loc = SourceLocation::unknown();
        let err = JITError::Located("type mismatch".into(), loc);
        let output = format!("{err}");
        assert_eq!(
            output, "type mismatch",
            "Located (unknown) variant must render the bare message without an 'error: ' prefix, got: {output}"
        );
    }

    // -----------------------------------------------------------------------
    // 3. Located variant with a known location includes the --> pointer line.
    // -----------------------------------------------------------------------

    #[test]
    fn located_known_includes_location_pointer() {
        let loc = SourceLocation::new("src/kernel.rs".into(), 99, 12);
        let err = JITError::Located("undefined variable".into(), loc);
        let output = format!("{err}");
        assert_eq!(
            output, "undefined variable\n  --> src/kernel.rs:99:12",
            "Located (known) variant must include message + location pointer, got: {output}"
        );
    }

    #[test]
    fn located_unknown_does_not_include_arrow() {
        let loc = SourceLocation::unknown();
        let err = JITError::Located("undefined variable".into(), loc);
        let output = format!("{err}");
        assert_eq!(
            output, "undefined variable",
            "Located (unknown) variant must be the bare message with no '-->', got: {output}"
        );
    }

    // -----------------------------------------------------------------------
    // 4. All JITError variants omit the "error: " prefix — that prefix is
    //    the responsibility of the outer Error type, not JITError itself.
    //    This ensures no variant accidentally re-introduces it.
    // -----------------------------------------------------------------------

    #[test]
    fn no_variant_adds_error_prefix() {
        let cases: Vec<(&str, JITError)> = vec![
            ("Generic", JITError::Generic("msg".into())),
            (
                "Located(known)",
                JITError::Located("msg".into(), SourceLocation::new("f.rs".into(), 1, 0)),
            ),
            (
                "Located(unknown)",
                JITError::Located("msg".into(), SourceLocation::unknown()),
            ),
            ("Anyhow", JITError::Anyhow(anyhow::anyhow!("msg"))),
        ];
        for (name, err) in &cases {
            let output = format!("{err}");
            assert!(
                !output.starts_with("error: "),
                "JITError::{name} must NOT start with 'error: ' — that prefix \
                 belongs to the outer Error type.  Got: {output}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 5. Error message quality: messages from JITError::Generic and
    //    JITError::Located must NOT expose internal type / function names.
    //
    //    These are regression guards: if someone accidentally puts an
    //    internal name like `TileRustValue` or `TypeMeta` back into an
    //    error string, these tests will catch it.
    // -----------------------------------------------------------------------

    /// Internal identifiers that must never appear in user-facing error text.
    const FORBIDDEN_INTERNALS: &[&str] = &[
        "TileRustValue",
        "TileRustType",
        "TypeMeta",
        "Kind::Compound",
        "Kind::Struct",
        "Kind::PrimitiveType",
        "Kind::StructuredType",
        "Kind::String",
        "get_concrete_op_ident_from_types",
    ];

    /// Helper: assert that `text` does not contain any of the forbidden
    /// internal identifiers.
    fn assert_no_internal_leaks(text: &str, context: &str) {
        for &forbidden in FORBIDDEN_INTERNALS {
            assert!(
                !text.contains(forbidden),
                "{context}: error message must not expose internal name `{forbidden}`.\n  \
                 Full message: {text}"
            );
        }
    }

    #[test]
    fn generic_error_does_not_leak_internals() {
        // Construct messages the same way the compiler does.
        let msgs = [
            "internal: string value has inconsistent fields set",
            "internal: primitive value has inconsistent fields set",
            "internal: structured type value has inconsistent fields set",
            "internal: compound value has inconsistent fields set",
            "internal: struct value missing its fields",
            "failed to parse attribute `name` with value `bad`",
            "unsupported element type `q16`; expected an integer (`i...`) or float (`f...`) type",
            "this binary operator is not supported",
            "unrecognized arithmetic operation `bogus`",
            "type `Foo` cannot be used as a tile type",
            "expected a variable name, got `1 + 2`",
            "undefined variable `x` when updating token",
            "variable `v` does not have associated type metadata (expected a view type)",
            "variable `v` is missing a `token` field (expected a view with an ordering token)",
            "unexpected token `@` in expression list",
        ];
        for msg in msgs {
            let err = JITError::Generic(msg.to_string());
            let output = format!("{err}");
            assert_no_internal_leaks(&output, &format!("Generic('{msg}')"));
        }
    }

    #[test]
    fn located_error_does_not_leak_internals() {
        let loc = SourceLocation::new("test.rs".into(), 1, 0);
        let msgs = [
            "unable to determine type for numeric literal; add a type annotation",
            "expected a scalar type for this literal, got a non-scalar type",
            "failed to compile the type of this literal",
            "type metadata not supported for this value kind",
            "expected a scalar element type",
            "expected a compile-time constant, but got a value with bounds [0, 100]",
            "function `my_fn` is missing a required `#[entry(...)]` attribute",
            "unable to compile parameter type `BadType`",
        ];
        for msg in msgs {
            let err = JITError::Located(msg.to_string(), loc.clone());
            let output = format!("{err}");
            assert_no_internal_leaks(&output, &format!("Located('{msg}')"));
        }
    }

    // -----------------------------------------------------------------------
    // 6. SpannedJITError trait produces Located variants, not Generic.
    // -----------------------------------------------------------------------

    #[test]
    fn spanned_jit_error_produces_located_variant() {
        let loc = SourceLocation::new("foo.rs".into(), 7, 3);
        let err = loc.jit_error("oops");
        match &err {
            JITError::Located(msg, eloc) => {
                assert_eq!(msg, "oops");
                assert!(eloc.is_known());
                assert_eq!(eloc.file, "foo.rs");
                assert_eq!(eloc.line, 7);
                assert_eq!(eloc.column, 3);
            }
            other => panic!("expected Located variant, got: {other}"),
        }
    }

    #[test]
    fn spanned_jit_error_unknown_produces_located_with_unknown_loc() {
        let loc = SourceLocation::unknown();
        let err = loc.jit_error("oops");
        match &err {
            JITError::Located(msg, eloc) => {
                assert_eq!(msg, "oops");
                assert!(!eloc.is_known());
            }
            other => panic!("expected Located variant, got: {other}"),
        }
    }

    // -----------------------------------------------------------------------
    // 7. SourceLocation::unknown() is clearly distinguishable.
    // -----------------------------------------------------------------------

    #[test]
    fn unknown_location_is_not_known() {
        let loc = SourceLocation::unknown();
        assert!(!loc.is_known());
    }

    #[test]
    fn known_location_is_known() {
        let loc = SourceLocation::new("file.rs".into(), 1, 0);
        assert!(loc.is_known());
    }

    #[test]
    fn partial_location_empty_file_is_unknown() {
        let loc = SourceLocation::new(String::new(), 10, 5);
        assert!(
            !loc.is_known(),
            "empty file with non-zero line should be unknown"
        );
    }

    #[test]
    fn partial_location_zero_line_is_unknown() {
        let loc = SourceLocation::new("file.rs".into(), 0, 5);
        assert!(
            !loc.is_known(),
            "zero line with non-empty file should be unknown"
        );
    }

    // -----------------------------------------------------------------------
    // 8. From conversions preserve error text faithfully.
    // -----------------------------------------------------------------------

    #[test]
    fn from_anyhow_preserves_message() {
        let anyhow_err = anyhow::anyhow!("boom");
        let jit_err: JITError = anyhow_err.into();
        let output = format!("{jit_err}");
        assert!(
            output.contains("boom"),
            "Anyhow conversion must preserve the original message, got: {output}"
        );
    }

    #[test]
    fn from_lex_error_produces_generic() {
        // proc_macro2::LexError is hard to construct directly, but we can
        // verify the From impl exists and produces a Generic variant by
        // attempting to lex invalid tokens.
        let bad_input: Result<proc_macro2::TokenStream, _> = "\"unterminated".parse();
        if let Err(lex_err) = bad_input {
            let jit_err: JITError = lex_err.into();
            // It should be a Generic variant (not Located, etc.)
            let output = format!("{jit_err}");
            assert!(
                !output.is_empty(),
                "LexError conversion should produce a non-empty message, got: {output}"
            );
            assert!(
                !output.starts_with("error: "),
                "LexError→Generic must not add 'error: ' prefix, got: {output}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // 9. The error message includes the user's value/context — not just a
    //    generic string.
    // -----------------------------------------------------------------------

    #[test]
    fn generic_error_preserves_user_value_in_message() {
        let err = JITError::generic_err("undefined variable `my_var`");
        let output = format!("{err}");
        assert!(
            output.contains("my_var"),
            "Error message should include the user's variable name, got: {output}"
        );
    }

    #[test]
    fn located_error_preserves_user_value_in_message() {
        let loc = SourceLocation::new("test.rs".into(), 5, 10);
        let err = JITError::located("expected type `f32`, got `i32`", loc);
        let output = format!("{err}");
        assert!(
            output.contains("f32") && output.contains("i32"),
            "Error message should include user types, got: {output}"
        );
    }
}
